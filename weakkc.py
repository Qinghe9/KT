"""Shared WeakKC protocol used by the proposed model and adapted baselines."""
from dataclasses import asdict, dataclass
from collections import defaultdict

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class WeakKCConfig:
    history_ratio: float = 0.70
    min_history: int = 16
    min_future_per_kc: int = 10
    positive_error_rate: float = 0.60
    negative_error_rate: float = 0.40
    recall_ks: tuple = (1, 3, 5, 10)

    def validate(self):
        if not 0 < self.history_ratio < 1:
            raise ValueError("history_ratio must be in (0, 1)")
        if self.min_history < 1 or self.min_future_per_kc < 1:
            raise ValueError("minimum supports must be positive")
        if not 0 <= self.negative_error_rate < self.positive_error_rate <= 1:
            raise ValueError("require 0 <= negative threshold < positive threshold <= 1")
        if not self.recall_ks or any(k < 1 for k in self.recall_ks):
            raise ValueError("recall_ks must contain positive integers")
        return self

    def to_dict(self):
        value = asdict(self)
        value["recall_ks"] = list(self.recall_ks)
        return value


def weak_kc_examples(student, config=WeakKCConfig()):
    """Yield (cut, kc, label) using future answers only for label creation."""
    config.validate()
    cut = int(len(student["r"]) * config.history_ratio)
    if cut < config.min_history:
        return
    future = defaultdict(list)
    for concepts, response in zip(student["c"][cut:], student["r"][cut:]):
        # One question remains one KT interaction, but contributes evidence to
        # every distinct atomic KC attached to it.
        for kc in set(concepts if isinstance(concepts, list) else [concepts]):
            future[kc].append(response)
    for kc in sorted(future):
        answers = future[kc]
        if len(answers) < config.min_future_per_kc:
            continue
        error_rate = 1.0 - float(np.mean(answers))
        if error_rate >= config.positive_error_rate:
            yield cut, kc, 1
        elif error_rate <= config.negative_error_rate:
            yield cut, kc, 0


class WeakKCHead(nn.Module):
    """The identical target-KC-conditioned head used in every experiment."""
    def __init__(self, state_dim, num_concepts, common_dim=64, dropout=0.2):
        super().__init__()
        self.state_projection = nn.Linear(state_dim, common_dim)
        self.concept_embedding = nn.Embedding(num_concepts, common_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(2 * common_dim, common_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(common_dim, 1),
        )

    def forward(self, state, target_concept):
        state = self.state_projection(state)
        concept = self.concept_embedding(target_concept)
        return self.classifier(torch.cat((state, concept), dim=-1)).squeeze(-1)


def weak_recall_metrics(rows, ks=(1, 3, 5, 10)):
    """Macro Recall@K over users having at least one labelled weak KC."""
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["uid"]].append(row)
    values = {int(k): [] for k in ks}
    candidate_counts = []
    for candidates in by_user.values():
        positives = sum(int(row["y"]) for row in candidates)
        if not positives:
            continue
        ranked = sorted(candidates, key=lambda row: -row["p"])
        candidate_counts.append(len(ranked))
        for k in values:
            values[k].append(sum(int(row["y"]) for row in ranked[:k]) / positives)
    result = {f"Recall@{k}": (float(np.mean(v)) if v else None) for k, v in values.items()}
    result.update(recall_users=len(candidate_counts),
                  mean_candidates=float(np.mean(candidate_counts)) if candidate_counts else None,
                  candidate_definition="future-supported, confidently labelled KCs")
    return result
