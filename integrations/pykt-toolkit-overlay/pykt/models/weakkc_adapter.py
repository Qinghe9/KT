"""Hidden-state adapters for the ten baseline models in the experiment."""
import torch
from .weakkc_head import last_valid_state


def state_dim(model):
    name=model.model_name
    if name in ('dkt','dkt+'): return model.hidden_size
    if name=='dkvmn': return model.dim_s
    if name=='sakt': return model.emb_size
    if name=='saint': return model.out.in_features
    if name=='dimkt': return model.emb_size
    if name in ('akt','simplekt','dtransformer','mockt'):
        first=next(layer for layer in model.out.modules() if isinstance(layer,torch.nn.Linear))
        return first.in_features
    raise KeyError(name)


def encode_history(model,batch):
    """Return one strict-history vector per diagnostic candidate."""
    q,c,r=batch['q'].long(),batch['c'].long(),batch['r'].long()
    lengths=batch['length'].long()
    name=model.model_name
    if name in ('dkt','dkt+'):
        _,states=model(c,r,qtest=True)
    elif name=='dkvmn':
        _,states=model(c,r,qtest=True)
    elif name=='sakt':
        query=torch.cat((c[:,1:],batch['target_c'][:,None]),dim=1)
        _,states=model(c,r,query,qtest=True)
    elif name=='saint':
        # Append the diagnostic query: SAINT prepends its own response start
        # token, so this position summarizes every observed response.
        qx=torch.cat((q,torch.zeros_like(q[:,:1])),1)
        cx=torch.cat((c,(batch['target_c']-1)[:,None]),1)
        _,states=model(qx,cx,r,qtest=True)
        return states[torch.arange(len(lengths),device=lengths.device),lengths]
    elif name=='akt':
        _,_,states=model(c,r,q,qtest=True)
    elif name=='simplekt':
        dcur={'qseqs':q[:,:-1],'cseqs':c[:,:-1],'rseqs':r[:,:-1],
              'shft_qseqs':q[:,1:],'shft_cseqs':c[:,1:],'shft_rseqs':r[:,1:]}
        _,states=model(dcur,qtest=True)
    elif name=='dtransformer':
        _,states,_,_,_,_=model.predict(c,r,q)
    elif name=='dimkt':
        raise RuntimeError('DIMKT WeakKC requires difficulty tensors; use encode_dimkt_history')
    elif name=='mockt':
        seq=lengths[:,None]
        _,_,states=model(seq,c,r,q,qtest=True)
    else: raise KeyError(f'unsupported WeakKC baseline: {name}')
    return last_valid_state(states,lengths)


def encode_dimkt_history(model,batch,skill_difficulty,question_difficulty):
    q,c,r=batch['q'].long(),batch['c'].long(),batch['r'].long(); lengths=batch['length'].long()
    sd=skill_difficulty[c]; qd=question_difficulty[q]
    _,states=model(q,c,sd,qd,r,q,c,sd,qd,qtest=True)
    return last_valid_state(states,lengths)
