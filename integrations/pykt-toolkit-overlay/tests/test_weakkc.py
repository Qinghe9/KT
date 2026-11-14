import csv
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pykt.datasets.weakkc_dataset import WeakKCDataset
from pykt.models.dkt import DKT
from pykt.models.weakkc_adapter import encode_history,state_dim
from pykt.models.weakkc_head import WeakKCHead


class WeakKCTests(unittest.TestCase):
    def test_atomic_multiconcept_labels_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'data.csv'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['fold','uid','questions','concepts','responses'])
                w.writeheader();w.writerow({'fold':0,'uid':'u','questions':','.join(map(str,range(20))),
                    'concepts':','.join(['0_1']*20),'responses':','.join(['1']*14+['0']*6)})
            ds=WeakKCDataset(path,{0},maxlen=20,min_history=2,min_future=2)
            self.assertEqual(len(ds),2)
            self.assertEqual({int(ds[i]['target_c']) for i in range(2)},{1,2})
            self.assertTrue(all(int(ds[i]['future_support'])==6 for i in range(2)))

    def test_dkt_adapter_and_shared_head(self):
        model=DKT(4,8);batch={'q':torch.zeros(2,5,dtype=torch.long),
            'c':torch.tensor([[1,2,1,0,0],[2,3,1,2,0]]),'r':torch.tensor([[1,0,1,0,0],[0,1,1,0,0]]),
            'length':torch.tensor([3,4]),'target_c':torch.tensor([1,2])}
        state=encode_history(model,batch)
        self.assertEqual(tuple(state.shape),(2,8))
        self.assertEqual(tuple(WeakKCHead(state_dim(model),4)(state,batch['target_c']).shape),(2,))

if __name__=='__main__': unittest.main()
