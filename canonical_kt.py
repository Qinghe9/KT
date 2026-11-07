"""Canonical question-level KT test candidates shared with pyKT evaluation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_kt_manifest(test_source, question_prediction_tsv, destination,
                      dataset="algebra2005", context=200):
    """Freeze pyKT's question-level candidate keys, independent of predictions."""
    test_source, question_prediction_tsv = Path(test_source), Path(question_prediction_tsv)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(100_000_000)
    original = list(csv.DictReader(test_source.open()))
    keys = {}
    local_positions = {}
    first_global_index = {}
    with question_prediction_tsv.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            orirow, qidx = int(row["orirow"]), int(row["qidx"])
            label = int(float(row["late_trues"]))
            key = (orirow, qidx)
            if key in keys and keys[key] != label:
                raise ValueError(f"conflicting label for {key}")
            keys[key] = label
            # pyKT writes qidx as a global question counter and may omit an
            # unsupported question.  Subtract the row-specific offset rather
            # than ranking outputs, so omissions do not shift later targets.
            first_global_index.setdefault(orirow, qidx)
            local_positions[key] = qidx - first_global_index[orirow] + 1
    records = []
    for orirow, qidx in sorted(keys):
        row = original[orirow]
        q = [int(x) for x in row["questions"].split(",")]
        c = [sorted(set(int(k) for k in value.split("_"))) for value in row["concepts"].split(",")]
        r = [int(x) for x in row["responses"].split(",")]
        t = ([int(x) for x in row["timestamps"].split(",")]
             if row.get("timestamps") else list(range(len(r))))
        local_qidx = local_positions[(orirow, qidx)]
        if not 0 < local_qidx < len(r) or r[local_qidx] != keys[(orirow, qidx)]:
            raise ValueError(f"invalid canonical KT key {(orirow, qidx)}")
        start = max(0, local_qidx - context)
        sample_id = hashlib.sha256(
            f"kt-canonical-v1|{dataset}|test|{row['uid']}|{orirow}|{qidx}".encode()
        ).hexdigest()[:24]
        records.append({"sample_id": sample_id, "uid": row["uid"], "orirow": orirow,
                        "qidx": qidx, "local_qidx": local_qidx,
                        "q": q[start:local_qidx], "c": c[start:local_qidx],
                        "r": r[start:local_qidx], "t": t[start:local_qidx],
                        "target_q": q[local_qidx], "target_cs": c[local_qidx],
                        "label": r[local_qidx]})
    path = destination / "test.jsonl"
    with path.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    manifest = {"protocol_version": "kt-canonical-v1", "dataset": dataset,
                "test_source": str(test_source.resolve()), "test_source_sha256": _sha256(test_source),
                "candidate_source": str(question_prediction_tsv.resolve()),
                "candidate_count": len(records), "context": context,
                "test_jsonl_sha256": _sha256(path)}
    manifest["protocol_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class CanonicalKTDataset(Dataset):
    def __init__(self, directory):
        self.directory = Path(directory); self.context = 200; self.weak = True
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        path = self.directory / "test.jsonl"
        if _sha256(path) != self.manifest["test_jsonl_sha256"]:
            raise ValueError("canonical KT candidate hash mismatch")
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        self.max_concepts = max(len(cs) for row in self.rows for cs in row["c"] + [row["target_cs"]])
        self.items = [(i, len(row["r"]), row["target_q"] + 1,
                       row["target_cs"][0] + 1, row["label"])
                      for i, row in enumerate(self.rows)]

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]; n = len(row["r"])
        q = np.zeros(self.context, np.int64); r = np.zeros(self.context, np.int64)
        gap = np.zeros(self.context, np.int64)
        c = np.zeros((self.context, self.max_concepts), np.int64)
        qv = np.asarray(row["q"], dtype=np.int64) + 1
        rv = np.asarray(row["r"], dtype=np.int64)
        q[:n], r[:n] = qv, rv + 1
        for pos, concepts in enumerate(row["c"]):
            shifted = [value + 1 for value in concepts]; c[pos, :len(shifted)] = shifted
        tv = np.asarray(row["t"], dtype=np.int64)
        if n:
            dt = np.maximum(0, np.diff(tv, prepend=tv[0])) / 1000
            gap[:n] = np.minimum(31, np.log2(1 + dt).astype(int)) + 1
        target_q = row["target_q"] + 1
        target_cs = [value + 1 for value in row["target_cs"]]
        sameq = qv == target_q
        samec = np.asarray([bool(set(cs) & set(row["target_cs"])) for cs in row["c"]])
        features=[]
        for mask in [np.ones(n,dtype=bool),samec,sameq]:
            values=rv[mask]
            features.extend([(values.sum()+1)/(len(values)+2),np.log1p(len(values))/5,
                             values[-1] if len(values) else .5,
                             values[-5:].mean() if len(values) else .5])
        return {"q":torch.tensor(q),"c":torch.tensor(c),"r":torch.tensor(r),
                "gap":torch.tensor(gap),"length":torch.tensor(n),
                "target_q":torch.tensor(target_q),"target_c":torch.tensor(target_cs[0]),
                "target_cs":torch.tensor(target_cs+[0]*(self.max_concepts-len(target_cs))),
                "y":torch.tensor(row["label"],dtype=torch.float32),
                "features":torch.tensor(features,dtype=torch.float32),"uid":row["uid"],
                "sample_id":row["sample_id"]}
