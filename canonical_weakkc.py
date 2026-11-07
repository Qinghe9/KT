"""Immutable WeakKC candidate protocol shared by every model family.

The manifest is generated from the unsegmented question-level pyKT files.  It
stores raw question/KC identifiers; model-specific datasets may adapt these IDs
to their embedding convention, but must not change sample membership, the
diagnosis cut, labels, or the raw history boundary.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROTOCOL_VERSION = "weakkc-canonical-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_id(dataset: str, split: str, uid: str, cut: int, kc: int) -> str:
    value = f"{PROTOCOL_VERSION}|{dataset}|{split}|{uid}|{cut}|{kc}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def build_canonical_manifest(train_valid_source, test_source, destination,
                             dataset="algebra2005", fold=0, history_ratio=.7,
                             min_history=16, min_future=10,
                             positive_threshold=.6, negative_threshold=.4,
                             max_history=200):
    """Create deterministic JSONL candidate files and a signed manifest."""
    train_valid_source, test_source = Path(train_valid_source), Path(test_source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(100_000_000)
    sources = {"train_valid": train_valid_source, "test": test_source}
    records = {"train": [], "valid": [], "test": []}

    for source_name, path in sources.items():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row_fold = int(row.get("fold", -1))
                if source_name == "test":
                    split = "test"
                else:
                    split = "valid" if row_fold == fold else "train"
                q = [int(x) for x in row["questions"].split(",")]
                cs = [sorted(set(int(k) for k in value.split("_")))
                      for value in row["concepts"].split(",")]
                r = [int(x) for x in row["responses"].split(",")]
                if row.get("timestamps"):
                    t = [int(x) for x in row["timestamps"].split(",")]
                else:
                    t = list(range(len(r)))
                if not (len(q) == len(cs) == len(r) == len(t)):
                    raise ValueError(f"unaligned canonical row uid={row['uid']}")
                cut = int(len(r) * history_ratio)
                if cut < min_history:
                    continue
                future = defaultdict(list)
                for concepts, answer in zip(cs[cut:], r[cut:]):
                    for kc in concepts:
                        future[kc].append(answer)
                start = max(0, cut - max_history)
                for kc, answers in sorted(future.items()):
                    if len(answers) < min_future:
                        continue
                    error_rate = 1.0 - sum(answers) / len(answers)
                    if error_rate >= positive_threshold:
                        label = 1
                    elif error_rate <= negative_threshold:
                        label = 0
                    else:
                        continue
                    records[split].append({
                        "sample_id": _sample_id(dataset, split, row["uid"], cut, kc),
                        "uid": row["uid"], "target_kc": kc, "label": label,
                        "future_support": len(answers), "future_error_rate": error_rate,
                        "sequence_length": len(r), "diagnosis_cut": cut,
                        "history_start": start, "history_end": cut,
                        "q": q[start:cut], "c": cs[start:cut],
                        "r": r[start:cut], "t": t[start:cut],
                    })

    for split, rows in records.items():
        rows.sort(key=lambda x: (str(x["uid"]), x["target_kc"], x["sample_id"]))
        with (destination / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "protocol_version": PROTOCOL_VERSION, "dataset": dataset, "fold": fold,
        "history_ratio": history_ratio, "min_history": min_history,
        "min_future_per_kc": min_future,
        "positive_error_rate": positive_threshold,
        "negative_error_rate": negative_threshold, "max_history_questions": max_history,
        "diagnosis_unit": "one candidate per (student, atomic KC)",
        "future_counting": "one response per distinct atomic KC per original question",
        "history_boundary": "strictly before floor(sequence_length * history_ratio)",
        "ambiguous_band": f"({negative_threshold}, {positive_threshold}) excluded",
        "train_valid_source": str(train_valid_source.resolve()),
        "test_source": str(test_source.resolve()),
        "train_valid_sha256": _sha256(train_valid_source),
        "test_sha256": _sha256(test_source),
        "split_counts": {split: len(rows) for split, rows in records.items()},
        "positive_counts": {split: sum(row["label"] for row in rows)
                            for split, rows in records.items()},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    file_hashes = {p.name: _sha256(p) for p in sorted(destination.glob("*.jsonl"))}
    manifest["candidate_file_sha256"] = file_hashes
    manifest["protocol_hash"] = hashlib.sha256(
        json.dumps({k: v for k, v in manifest.items() if k != "protocol_hash"},
                   sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


class CanonicalWeakKCDataset(Dataset):
    """Adapter from the immutable manifest to the proposed model input format."""
    def __init__(self, directory, split, context=200, num_concepts=1):
        self.directory, self.split = Path(directory), split
        self.context = int(context)
        self.weak = True
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        path = self.directory / f"{split}.jsonl"
        expected = self.manifest["candidate_file_sha256"][path.name]
        if _sha256(path) != expected:
            raise ValueError(f"canonical candidate hash mismatch: {path}")
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        self.max_concepts = max((len(cs) for row in self.rows for cs in row["c"]), default=1)
        self.num_concepts = num_concepts
        self.items = [(i, len(row["r"]), 0, row["target_kc"] + 1, row["label"])
                      for i, row in enumerate(self.rows)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        n = min(len(row["r"]), self.context)
        q = np.zeros(self.context, dtype=np.int64)
        r = np.zeros(self.context, dtype=np.int64)
        gap = np.zeros(self.context, dtype=np.int64)
        c = np.zeros((self.context, self.max_concepts), dtype=np.int64)
        q_values = np.asarray(row["q"][-n:], dtype=np.int64) + 1
        r_values = np.asarray(row["r"][-n:], dtype=np.int64)
        c_values = row["c"][-n:]
        t_values = np.asarray(row["t"][-n:], dtype=np.int64)
        q[:n], r[:n] = q_values, r_values + 1
        for pos, concepts in enumerate(c_values):
            shifted = [value + 1 for value in concepts]
            c[pos, :len(shifted)] = shifted
        if n:
            dt = np.maximum(0, np.diff(t_values, prepend=t_values[0])) / 1000
            gap[:n] = np.minimum(31, np.log2(1 + dt).astype(int)) + 1
        target = row["target_kc"] + 1
        hist_r = r_values
        samec = np.asarray([row["target_kc"] in concepts for concepts in c_values])
        features = []
        for mask in [np.ones(n, dtype=bool), samec, np.zeros(n, dtype=bool)]:
            values = hist_r[mask]
            features.extend([(values.sum() + 1) / (len(values) + 2), np.log1p(len(values)) / 5,
                             values[-1] if len(values) else .5,
                             values[-5:].mean() if len(values) else .5])
        return {
            "q": torch.tensor(q), "c": torch.tensor(c), "r": torch.tensor(r),
            "gap": torch.tensor(gap), "length": torch.tensor(n),
            "target_q": torch.tensor(0), "target_c": torch.tensor(target),
            "y": torch.tensor(row["label"], dtype=torch.float32),
            "target_cs": torch.tensor([target] + [0] * (self.max_concepts - 1)),
            "features": torch.tensor(features, dtype=torch.float32), "uid": row["uid"],
            "sample_id": row["sample_id"],
            "future_support": torch.tensor(row["future_support"]),
            "future_error_rate": torch.tensor(row["future_error_rate"], dtype=torch.float32),
        }
