"""Leakage-free atomic-KC diagnostic samples built from question-level CSVs."""
import csv
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class WeakKCDataset(Dataset):
    def __init__(self, path, folds, maxlen=200, history_ratio=.7, min_history=16,
                 min_future=10, positive_threshold=.6, negative_threshold=.4):
        csv.field_size_limit(100_000_000)
        self.samples=[]; self.maxlen=maxlen
        with open(path,encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if int(row.get('fold',-1)) not in folds: continue
                q=[int(x) for x in row['questions'].split(',')]
                cs=[sorted(set(int(k) for k in x.split('_'))) for x in row['concepts'].split(',')]
                r=[int(x) for x in row['responses'].split(',')]
                if not (len(q)==len(cs)==len(r)): raise ValueError(f"unaligned row uid={row['uid']}")
                cut=int(len(r)*history_ratio)
                if cut<min_history: continue
                future=defaultdict(list)
                for concepts,answer in zip(cs[cut:],r[cut:]):
                    for kc in concepts: future[kc].append(answer)
                # pyKT baselines consume atomic concept streams. Multi-KC questions
                # are expanded only in the history encoder; label counting above
                # de-duplicates KCs per original question.
                hq=[];hc=[];hr=[]
                for question,concepts,answer in zip(q[:cut],cs[:cut],r[:cut]):
                    for kc in concepts:
                        hq.append(question);hc.append(kc);hr.append(answer)
                hq,hc,hr=hq[-maxlen:],hc[-maxlen:],hr[-maxlen:]
                for kc,answers in sorted(future.items()):
                    if len(answers)<min_future: continue
                    error=1-sum(answers)/len(answers)
                    if error>=positive_threshold: label=1
                    elif error<=negative_threshold: label=0
                    else: continue
                    # +1 is used only by the diagnostic head so index 0 remains
                    # padding. Native pyKT history IDs stay unchanged.
                    self.samples.append((row['uid'],hq,hc,hr,kc+1,label,len(answers),error))

    def __len__(self): return len(self.samples)

    def __getitem__(self,index):
        uid,q,c,r,target,label,support,error=self.samples[index]; n=len(r)
        def padded(values): return torch.tensor(values+[0]*(self.maxlen-n),dtype=torch.long)
        return dict(uid=uid,q=padded(q),c=padded(c),r=padded(r),length=torch.tensor(n),
                    target_c=torch.tensor(target),label=torch.tensor(label,dtype=torch.float32),
                    future_support=torch.tensor(support),future_error_rate=torch.tensor(error,dtype=torch.float32))
