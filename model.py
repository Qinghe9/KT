"""Knowledge-conditioned spectral mixture; FFT is computed on strict prefixes only."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from weakkc import WeakKCHead


class CausalCNN(nn.Module):
    """Multi-scale causal depthwise CNN; position t never sees positions > t."""
    def __init__(self,d,kernels=(2,3,5)):
        super().__init__()
        self.kernels=kernels
        self.convs=nn.ModuleList([nn.Conv1d(d,d,k,groups=d) for k in kernels])
        self.mix=nn.Linear(len(kernels)*d,d)

    def forward(self,x):
        xt=x.transpose(1,2)
        ys=[]
        for k,conv in zip(self.kernels,self.convs):
            # Left padding only. Symmetric Conv1d padding would leak later states.
            ys.append(F.gelu(conv(F.pad(xt,(k-1,0))).transpose(1,2)))
        return self.mix(torch.cat(ys,-1))


class SpectralKT(nn.Module):
    def __init__(self,nq,nc,context=32,d=32,topk=2,history_skip=True,spectral=True,item_prior=False,
                 kc_conditioning=True,uniform_fusion=False,disabled_experts=()):
        super().__init__()
        self.context,self.topk,self.spectral=context,topk,spectral
        self.kc_conditioning,self.uniform_fusion=kc_conditioning,uniform_fusion
        self.q=nn.Embedding(nq,d,padding_idx=0); self.c=nn.Embedding(nc,d,padding_idx=0)
        self.r=nn.Embedding(3,d,padding_idx=0); self.gap=nn.Embedding(33,d,padding_idx=0)
        self.pos=nn.Embedding(context,d)
        self.norm=nn.LayerNorm(d); self.condition=nn.Linear(d,d,bias=False)
        self.value=nn.Linear(d,d,bias=False)
        # Interpolate a fixed frequency grid to the actual valid prefix length.
        self.filter_real=nn.Parameter(torch.ones(4,context//2+1,d))
        self.filter_imag=nn.Parameter(torch.zeros(4,context//2+1,d))
        self.gamma=nn.Parameter(torch.full((4,),.1))
        self.filtered_norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.attention=nn.MultiheadAttention(d,4,batch_first=True)
        self.gru=nn.GRU(d,d,batch_first=True)
        self.cnn=CausalCNN(d)
        # Official state-spaces/mamba implementation (mamba_ssm), not a local approximation.
        self.mamba=Mamba(d_model=d,d_state=16,d_conv=4,expand=2)
        self.expert_names=('attention','gru','cnn','mamba_official')
        unknown=set(disabled_experts)-set(self.expert_names)
        if unknown: raise ValueError(f'unknown disabled experts: {sorted(unknown)}')
        active=[i for i,name in enumerate(self.expert_names) if name not in disabled_experts]
        if not active: raise ValueError('at least one expert must remain active')
        # Runtime architecture selector; excluded from checkpoints so older full-model
        # checkpoints remain loadable for the two-stage diagnostic experiment.
        self.register_buffer('active_experts',torch.tensor(active,dtype=torch.long),persistent=False)
        self.router=nn.Sequential(nn.Linear(2*d+5,d),nn.GELU(),nn.Linear(d,4))
        self.kt=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Linear(d,1))
        # Identical 64-diagnostic-dimension head used by every adapted baseline.
        self.weak=WeakKCHead(d,nc,common_dim=64,dropout=.2)
        self.history_skip=history_skip
        self.stats_kt=nn.Sequential(nn.Linear(12,d),nn.GELU(),nn.Linear(d,1))
        self.item_bias=nn.Embedding(nq,1) if item_prior else None

    def concept_mean(self, ids):
        emb=self.c(ids)
        mask=ids.ne(0).unsqueeze(-1)
        return (emb*mask).sum(-2)/mask.sum(-2).clamp_min(1)

    def spectrum(self,z,length):
        b,l,d=z.shape
        filtered=z.unsqueeze(1).expand(-1,4,-1,-1).clone()
        stats=z.new_zeros((b,5))
        # Group by effective length: padding is never included in FFT or its mean.
        for n in length.unique().tolist():
            idx=(length==n).nonzero(as_tuple=True)[0]
            seq=z[idx,:n]; centered=seq-seq.mean(1,keepdim=True)
            freq=torch.fft.rfft(centered,dim=1,norm='ortho')
            power=freq.abs().square().sum(-1)[:,1:]
            if power.shape[1]:
                total=power.sum(-1,keepdim=True); valid=total[:,0]>1e-8
                prob=power/total.clamp_min(1e-8)
                f=torch.fft.rfftfreq(n,device=z.device)[1:]
                bands=[(f<=1/6),(f>1/6)&(f<=1/3),f>1/3]
                for k,band in enumerate(bands): stats[idx,k]=prob[:,band].sum(-1)
                stats[idx,3]=-(prob*prob.clamp_min(1e-8).log()).sum(-1)/max(math.log(len(f)),1)
                stats[idx,4]=valid.float()
            if self.spectral:
                real=F.interpolate(self.filter_real.transpose(1,2),size=freq.shape[1],mode='linear',align_corners=True).transpose(1,2)
                imag=F.interpolate(self.filter_imag.transpose(1,2),size=freq.shape[1],mode='linear',align_corners=True).transpose(1,2)
                out=torch.fft.irfft(freq[:,None]*torch.complex(real,imag)[None],n=n,dim=2,norm='ortho')
                filtered[idx,:,:n]=seq[:,None]+self.gamma[None,:,None,None]*out
        if not self.spectral:
            stats=stats*0  # Ablation also removes frequency information from routing.
        return filtered,stats

    def forward(self,b,weak=False):
        mask=b['q']!=0; length=b['length']; size=len(length)
        target=self.c(b['target_c']) if weak else self.concept_mean(b['target_cs'])
        pos=torch.arange(mask.shape[1],device=mask.device)
        x=self.norm(self.q(b['q'])+self.concept_mean(b['c'])+self.r(b['r'])+self.gap(b['gap'])+self.pos(pos))
        if self.kc_conditioning:
            gate=torch.sigmoid((self.condition(target)[:,None]*x).sum(-1)/math.sqrt(x.shape[-1]))
            z=self.norm(x+gate[:,:,None]*self.value(x))*mask[:,:,None]
        else:
            # Conditioning ablation: history encoding is independent of target KC.
            z=x*mask[:,:,None]
        inputs,stats=self.spectrum(z,length)
        pooled=z.sum(1)/length[:,None]
        logits=self.router(torch.cat([pooled,target,stats],-1))
        active=self.active_experts
        active_logits=logits[:,active]
        if self.uniform_fusion:
            active_sparse=torch.full_like(active_logits,1.0/len(active))
            active_dense=active_sparse
        else:
            active_dense=active_logits.softmax(-1)
            k=min(self.topk,len(active))
            top=active_logits.topk(k,-1).indices
            active_sparse=torch.full_like(active_logits,-torch.inf).scatter(
                1,top,active_logits.gather(1,top)).softmax(-1)
        # Some CUDA autocast kernels return fp16 softmax values while the router
        # logits stay fp32. Scatter/index_add require exact dtype agreement.
        active_dense=active_dense.to(logits.dtype)
        active_sparse=active_sparse.to(logits.dtype)
        dense=torch.zeros_like(logits).scatter(1,active[None].expand(size,-1),active_dense)
        sparse=torch.zeros_like(logits).scatter(1,active[None].expand(size,-1),active_sparse)
        result=z.new_zeros((size,z.shape[-1]))
        for j in range(4):
            selected=(sparse[:,j]>0).nonzero(as_tuple=True)[0]
            if not len(selected): continue
            u=self.filtered_norm[j](inputs[selected,j]); m=mask[selected]; lens=length[selected]
            last=u[torch.arange(len(u),device=u.device),lens-1]
            if j==0:
                h=self.attention(target[selected,None],u,u,key_padding_mask=~m,need_weights=False)[0][:,0]
            elif j==1:
                enc,_=self.gru(u); h=enc[torch.arange(len(u),device=u.device),lens-1]
            elif j==2:
                enc=self.cnn(u); h=enc[torch.arange(len(u),device=u.device),lens-1]
            else:
                enc=self.mamba(u); h=enc[torch.arange(len(u),device=u.device),lens-1]
            contribution=(h*sparse[selected,j,None]).to(result.dtype)
            result=result.index_add(0,selected,contribution)
        if weak:
            pred=self.weak(result,b['target_c'])
        else:
            pred=self.kt(torch.cat([result,self.q(b['target_q']),target],-1)).squeeze(-1)
            if self.history_skip: pred=pred+self.stats_kt(b['features']).squeeze(-1)
            if self.item_bias is not None: pred=pred+self.item_bias(b['target_q']).squeeze(-1)
        # Dense probabilities supply gradients to experts outside top-k as well.
        balance=((dense[:,active].mean(0)-1.0/len(active))**2).sum()
        return pred,balance,sparse
