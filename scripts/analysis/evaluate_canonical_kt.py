#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from canonical_kt import CanonicalKTDataset
from data import LengthBucketBatchSampler
from model import SpectralKT
from train import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest-dir',required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch',type=int,default=2048)
    args=parser.parse_args()
    meta=json.loads((Path(args.data)/'manifest.json').read_text())
    config=json.loads(Path(args.config).read_text())
    ds=CanonicalKTDataset(args.manifest_dir)
    loader=DataLoader(ds,batch_sampler=LengthBucketBatchSampler(ds,args.batch,shuffle=False),
                      num_workers=1,pin_memory=args.device.startswith('cuda'))
    model=SpectralKT(meta['num_q'],meta['num_c'],config['context'],config['dim'],
                     topk=config['topk'],history_skip=not config.get('no_history_skip',False),
                     spectral=not config.get('no_spectrum',False),item_prior=config.get('item_prior',False),
                     kc_conditioning=not config.get('no_kc_conditioning',False),
                     uniform_fusion=config.get('uniform_fusion',False),
                     disabled_experts=tuple(config.get('disable_expert',[]))).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint,map_location=args.device,weights_only=True))
    result,rows=evaluate(model,loader,args.device)
    result['protocol_hash']=ds.manifest['protocol_hash']
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'canonical_kt_metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'canonical_kt_predictions.csv').open('w') as handle:
        fields=['sample_id','uid','concepts','y','p','future_support','future_error_rate']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
