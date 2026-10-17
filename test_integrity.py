import unittest
import inspect
import torch
from data import PrefixDataset
from model import SpectralKT, CausalCNN
from mamba_ssm import Mamba
from weakkc import WeakKCConfig, WeakKCHead, weak_kc_examples, weak_recall_metrics


class Integrity(unittest.TestCase):
    def test_official_mamba_and_causal_cnn(self):
        self.assertIn('site-packages/mamba_ssm',inspect.getfile(Mamba))
        torch.manual_seed(2)
        cnn=CausalCNN(8).eval(); x=torch.randn(2,8,8)
        before=cnn(x); changed=x.clone(); changed[:,4:]=1000
        after=cnn(changed)
        self.assertTrue(torch.allclose(before[:,:4],after[:,:4],atol=1e-6))

    def test_target_and_future_excluded(self):
        s=dict(uid='1',split='train',q=[1]*40,c=[[1]]*40,r=[0]*40,t=list(range(40)))
        ds=PrefixDataset([s],'train',16)
        before=ds[4]
        s['r'][5:]=[1]*35
        after=PrefixDataset([s],'train',16)[4]
        for key in ['q','c','r','gap','features','length']:
            self.assertTrue(torch.equal(before[key],after[key]),key)
        self.assertNotEqual(before['y'].item(),after['y'].item())

    def test_fft_padding_and_backward(self):
        torch.manual_seed(1);torch.set_num_threads(1)
        m=SpectralKT(5,5,8,16)
        z=torch.randn(2,8,16,requires_grad=True);length=torch.tensor([3,7])
        a,stats=m.spectrum(z,length)
        mutated=z.detach().clone();mutated[0,3:]=1000
        b,_=m.spectrum(mutated,length)
        self.assertTrue(torch.allclose(a[0,:,:3],b[0,:,:3]))
        (a.sum()+stats.sum()).backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_sparse_and_singleton(self):
        s=dict(uid='1',split='train',q=[1]*8,c=[[1]]*8,r=[0,1]*4,t=list(range(8)))
        b=PrefixDataset([s],'train',8)[0]
        b={k:v.unsqueeze(0) if isinstance(v,torch.Tensor) else [v] for k,v in b.items()}
        m=SpectralKT(3,3,8,16);pred,balance,g=m(b)
        self.assertEqual(int((g>0).sum()),2)
        self.assertTrue(torch.isfinite(pred).all())
        (pred.sum()+balance).backward()

    def test_ablation_architecture_switches(self):
        # Disable Mamba here so this structural test also runs on CPU; the
        # official CUDA Mamba path is exercised by the training smoke runs.
        s=dict(uid='1',split='train',q=[1]*8,c=[[1]]*8,r=[0,1]*4,t=list(range(8)))
        b=PrefixDataset([s],'train',8)[0]
        b={k:v.unsqueeze(0) if isinstance(v,torch.Tensor) else [v] for k,v in b.items()}
        for kwargs in (
            dict(kc_conditioning=False,disabled_experts=('mamba_official',)),
            dict(uniform_fusion=True,topk=4,disabled_experts=('mamba_official',)),
            dict(disabled_experts=('attention','mamba_official')),
            dict(disabled_experts=('gru','mamba_official')),
            dict(disabled_experts=('cnn','mamba_official')),
        ):
            model=SpectralKT(3,3,8,16,**kwargs)
            pred,balance,gates=model(b)
            self.assertTrue(torch.isfinite(pred).all())
            self.assertTrue(torch.isfinite(balance))
            self.assertEqual(int((gates>0).sum()),min(model.topk,len(model.active_experts))
                             if not model.uniform_fusion else len(model.active_experts))

    def test_weak_labels_do_not_change_inputs(self):
        s=dict(uid='1',split='train',q=[1]*60,c=[[1]]*60,r=[0]*60,t=list(range(60)))
        before=PrefixDataset([s],'train',16,weak=True)[0]
        s['r'][42:]=[1]*18
        after=PrefixDataset([s],'train',16,weak=True)[0]
        for key in ['q','c','r','gap','features','length']:
            self.assertTrue(torch.equal(before[key],after[key]))
        self.assertNotEqual(before['y'].item(),after['y'].item())

    def test_shared_weak_protocol_boundaries_and_recall(self):
        cfg=WeakKCConfig(min_history=2,min_future_per_kc=2)
        s=dict(uid='u',c=[1,1,1,1,1,1,2,2,2,2],r=[1,1,1,1,0,0,1,1,1,1])
        examples=list(weak_kc_examples(s,cfg))
        self.assertEqual(examples,[(7,2,0)])
        rows=[dict(uid='a',y=1,p=.9),dict(uid='a',y=0,p=.8),
              dict(uid='a',y=1,p=.1),dict(uid='b',y=0,p=.7)]
        score=weak_recall_metrics(rows,(1,3))
        self.assertEqual(score['recall_users'],1)
        self.assertAlmostEqual(score['Recall@1'],.5)
        self.assertAlmostEqual(score['Recall@3'],1.)

    def test_shared_head_shape_and_gradients(self):
        head=WeakKCHead(12,8,common_dim=6)
        state=torch.randn(4,12); concept=torch.tensor([1,2,3,4])
        output=head(state,concept)
        self.assertEqual(tuple(output.shape),(4,))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_no_duplicate_targets_or_cross_block_history(self):
        s=dict(uid='1',split='train',q=[1]*70,c=[[1]]*70,r=[0]*70,t=list(range(70)))
        ds=PrefixDataset([s],'train',16)
        self.assertEqual(len(ds.items),len(set((i,t) for i,t,*_ in ds.items)))
        self.assertTrue(all(t%16!=0 for _,t,*_ in ds.items))
        for idx,(_,t,*_) in enumerate(ds.items):
            self.assertEqual(ds[idx]['length'].item(),t%16)


if __name__=='__main__':unittest.main()
