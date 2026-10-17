"""Student-disjoint Junyi prototype; prefixes never contain their target answer."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from weakkc import WeakKCConfig, weak_kc_examples


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def prepare(source, destination, users=1600, seed=1024, cap=200):
    """Deterministic student sample from a pyKT question-level sequence CSV.

    A multi-concept question remains one interaction.  Its sorted concept-set token
    is mapped to one dense ID, avoiding the answer duplication caused by pyKT's
    concept-expanded sequence file.
    """
    source, destination = Path(source), Path(destination)
    csv.field_size_limit(100_000_000)
    candidates = []
    with source.open() as f:
        for row in csv.DictReader(f):
            uid = row['uid']
            rank = hashlib.sha256(f'{seed}:{uid}'.encode()).hexdigest()
            candidates.append((rank, uid))
    selected = [uid for _, uid in sorted(candidates)[:users]]
    if len(set(selected)) != len(selected):
        raise ValueError('Input must contain one original sequence per student')
    a, b = int(len(selected)*.8), int(len(selected)*.9)
    membership = {uid: ('train' if i < a else 'valid' if i < b else 'test')
                  for i, uid in enumerate(selected)}
    raw = []
    with source.open() as f:
        for row in csv.DictReader(f):
            if row['uid'] not in membership:
                continue
            q = [int(x)+1 for x in row['questions'].split(',')][:cap]
            # Preserve atomic KCs. A question is never duplicated when it has
            # multiple concepts; the model averages these embeddings.
            c = [sorted(set(map(int,x.split('_')))) for x in row['concepts'].split(',')][:cap]
            r = [int(x) for x in row['responses'].split(',')][:cap]
            if row.get('timestamps'):
                t = [int(x) for x in row['timestamps'].split(',')][:cap]
            else:
                # Some ASSIST releases have only an ordered sequence, no wall-clock time.
                # Unit-spaced synthetic timestamps make the gap feature constant and do
                # not invent response-dependent information.
                t = list(range(len(r)))
            assert len(q) == len(c) == len(r) == len(t)
            raw.append(dict(uid=row['uid'], split=membership[row['uid']], q=q,c=c,r=r,t=t))
    atomic=sorted({x for s in raw for cs in s['c'] for x in cs})
    concept_vocab={token:i+1 for i,token in enumerate(atomic)}
    output=[]; nq=0
    for s in raw:
        nq=max(nq,max(s['q'])); s['c']=[[concept_vocab[x] for x in cs] for cs in s['c']]; output.append(s)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'students.json').open('w') as f:
        json.dump(output, f)
    meta = dict(source=str(source.resolve()), source_bytes=source.stat().st_size,source_sha256=file_sha256(source),
                users=len(output), seed=seed, cap=cap, num_q=nq+1, num_c=len(concept_vocab)+1,
                split='student-disjoint 80/10/10; hash ordered',
                sampling='hash independent of answers; first cap interactions per student',
                max_concepts=max(len(cs) for s in output for cs in s['c']),
                concept_unit='atomic KCs; masked mean per question; no answer duplication')
    (destination/'manifest.json').write_text(json.dumps(meta, indent=2))
    return meta


def prepare_fold(train_valid_source, test_source, destination, fold=0, users=0, seed=1024, cap=0):
    """Build a pyKT-compatible one-fold dataset without reading response labels for sampling.

    `fold` rows form validation, other train_valid rows form training, and the
    separately preprocessed question-level test file remains held out.
    """
    train_valid_source, test_source = Path(train_valid_source), Path(test_source)
    destination = Path(destination); csv.field_size_limit(100_000_000)

    def selected_uids(path, split):
        ranked=[]
        with path.open() as f:
            for row in csv.DictReader(f):
                if split=='train' and int(row['fold'])==fold: continue
                if split=='valid' and int(row['fold'])!=fold: continue
                uid=row['uid']; rank=hashlib.sha256(f'{seed}:{split}:{uid}'.encode()).hexdigest()
                ranked.append((rank,uid))
        if users>0:
            # Keep the requested total approximately 80:10:10.
            limit={'train':int(users*.8),'valid':max(1,int(users*.1)),'test':max(1,int(users*.1))}[split]
            ranked=sorted(ranked)[:limit]
        return {uid for _,uid in ranked}

    chosen={
        'train':selected_uids(train_valid_source,'train'),
        'valid':selected_uids(train_valid_source,'valid'),
        'test':selected_uids(test_source,'test'),
    }
    raw=[]

    def ingest(path, allowed, split):
        with path.open() as f:
            for row in csv.DictReader(f):
                if row['uid'] not in allowed: continue
                q=[int(x)+1 for x in row['questions'].split(',')]
                c=[sorted(set(map(int,x.split('_')))) for x in row['concepts'].split(',')]
                r=[int(x) for x in row['responses'].split(',')]
                if row.get('timestamps'): t=[int(x) for x in row['timestamps'].split(',')]
                else: t=list(range(len(r)))
                if cap>0: q,c,r,t=q[:cap],c[:cap],r[:cap],t[:cap]
                if not q: continue
                assert len(q)==len(c)==len(r)==len(t)
                raw.append(dict(uid=row['uid'],split=split,q=q,c=c,r=r,t=t))

    ingest(train_valid_source,chosen['train'],'train')
    ingest(train_valid_source,chosen['valid'],'valid')
    ingest(test_source,chosen['test'],'test')
    atomic=sorted({x for s in raw for cs in s['c'] for x in cs})
    concept_vocab={token:i+1 for i,token in enumerate(atomic)}
    nq=0
    for s in raw:
        nq=max(nq,max(s['q'])); s['c']=[[concept_vocab[x] for x in cs] for cs in s['c']]
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'students.json').open('w') as f: json.dump(raw,f)
    counts={split:sum(s['split']==split for s in raw) for split in chosen}
    meta=dict(train_valid_source=str(train_valid_source.resolve()),test_source=str(test_source.resolve()),
              train_valid_sha256=file_sha256(train_valid_source),test_sha256=file_sha256(test_source),
              fold=fold,users=len(raw),split_counts=counts,seed=seed,cap=cap,
              num_q=nq+1,num_c=len(concept_vocab)+1,
              max_concepts=max(len(cs) for s in raw for cs in s['c']),
              split='pyKT one-fold: folds!=selected train, selected fold valid, external test',
              sampling='all students' if users<=0 else 'hash sample independent of responses',
              concept_unit='atomic KCs; masked mean per question; no answer duplication')
    (destination/'manifest.json').write_text(json.dumps(meta,indent=2))
    return meta


class PrefixDataset(Dataset):
    def __init__(self, students, split, context=32, weak=False, weak_config=None):
        self.students = [s for s in students if s['split']==split]
        self.context, self.weak = context, weak
        self.weak_config = weak_config or WeakKCConfig()
        self.max_concepts=max((len(cs) for s in self.students for cs in s['c']),default=1)
        self.items = []
        for i,s in enumerate(self.students):
            if weak:
                for t,c,y in weak_kc_examples(s,self.weak_config):
                    self.items.append((i,t,0,c,y))
            else:
                # Non-overlapping context blocks. Each target scored exactly once;
                # each input is a strict prefix within its block, not a sliding test file.
                for t in range(1,len(s['r'])):
                    if t % context:
                        self.items.append((i,t,s['q'][t],s['c'][t][0],s['r'][t]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i,t,tq,tc,y = self.items[index]
        s = self.students[i]
        start = max(0,t-self.context) if self.weak else (t//self.context)*self.context
        n = t-start
        q,r,gap = [np.zeros(self.context,dtype=np.int64) for _ in range(3)]
        c=np.zeros((self.context,self.max_concepts),dtype=np.int64)
        q[:n],r[:n] = s['q'][start:t],np.array(s['r'][start:t])+1
        for row,concepts in enumerate(s['c'][start:t]): c[row,:len(concepts)]=concepts
        dt = np.maximum(0,np.diff(s['t'][start:t],prepend=s['t'][start]))/1000
        gap[:n] = np.minimum(31,np.log2(1+dt).astype(int))+1
        hist_r = np.array(s['r'][start:t]); hist_q=np.array(s['q'][start:t])
        target_cs=[tc] if self.weak else s['c'][t]
        sameq=hist_q==tq
        samec=np.array([bool(set(cs)&set(target_cs)) for cs in s['c'][start:t]])
        # Smoothed historical statistics use ONLY the same strict input prefix.
        features=[]
        for mask in [np.ones(n,dtype=bool),samec,sameq]:
            vals=hist_r[mask]
            features.extend([(vals.sum()+1)/(len(vals)+2), np.log1p(len(vals))/5,
                             vals[-1] if len(vals) else .5,
                             vals[-5:].mean() if len(vals) else .5])
        return dict(q=torch.tensor(q), c=torch.tensor(c), r=torch.tensor(r),
                    gap=torch.tensor(gap), length=torch.tensor(n), target_q=torch.tensor(tq),
                    target_c=torch.tensor(tc), y=torch.tensor(y,dtype=torch.float32),
                    target_cs=torch.tensor(target_cs+[0]*(self.max_concepts-len(target_cs))),
                    features=torch.tensor(features,dtype=torch.float32), uid=s['uid'])
