"""Common diagnostic head for fair WeakKC adaptations of pyKT encoders.

The base model must return its final valid history state.  Each baseline owns a
projection because encoder dimensions differ; the downstream head is otherwise
identical.  This module deliberately does not reuse weights across models.
"""
import torch
from torch import nn


class WeakKCHead(nn.Module):
    def __init__(self, state_dim, num_concepts, common_dim=64, dropout=0.2):
        super().__init__()
        self.state_projection = nn.Linear(state_dim, common_dim)
        self.concept_embedding = nn.Embedding(num_concepts + 1, common_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(common_dim, 1),
        )

    def forward(self, history_state, target_concept):
        state = self.state_projection(history_state)
        concept = self.concept_embedding(target_concept)
        return self.classifier(torch.cat((state, concept), dim=-1)).squeeze(-1)


def last_valid_state(sequence_states, lengths):
    """Select each student's last non-padding history state without future use."""
    index = (lengths - 1).clamp_min(0)
    return sequence_states[torch.arange(len(index), device=index.device), index]
