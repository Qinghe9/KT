"""Predict weak-KC risk using only a supplied student's observed prefix."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from data import PrefixDataset
from model import SpectralKT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--student-uid',required=True)
    p.add_argument('--at',type=int,required=True,help='number of observed interactions')
    args=p.parse_args();torch.set_num_threads(2)
    root=Path(args.run);cfg=json.loads((root/'config.json').read_text())
    data=Path(cfg['data']);meta=json.loads((data/'manifest.json').read_text())
    students=json.loads((data/'students.json').read_text())
    s=next(s for s in students if s['uid']==args.student_uid)
    if not 1<=args.at<=len(s['r']): raise ValueError('at outside observed sequence')
    # Physically discard every future interaction before building inputs.
    s={**s,**{k:s[k][:args.at] for k in ['q','c','r','t']}}
    ds=PrefixDataset([s],s['split'],cfg['context'],weak=True)
    ds.items=[(0,args.at,0,c,0) for c in sorted(set(s['c']))]
    model=SpectralKT(meta['num_q'],meta['num_c'],cfg['context'],cfg['dim'],
        history_skip=not cfg['no_history_skip'],spectral=not cfg['no_spectrum'],
        item_prior=cfg.get('item_prior',False))
    model.load_state_dict(torch.load(root/'best.pt',map_location='cpu',weights_only=True));model.eval()
    rows=[]
    with torch.no_grad():
        for b in DataLoader(ds,batch_size=256):
            logits,_,gates=model(b,weak=True)
            for c,risk,g in zip(b['target_c'].tolist(),logits.sigmoid().tolist(),gates.tolist()):
                rows.append(dict(concept=c,weak_risk=risk,observed_count=s['c'].count(c),expert_weights=g))
    print(json.dumps(dict(uid=s['uid'],observed_interactions=args.at,
                         interpretation='future error risk, not calibrated mastery probability',
                         concepts=sorted(rows,key=lambda r:-r['weak_risk'])),indent=2))


if __name__=='__main__':main()
