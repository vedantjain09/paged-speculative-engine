"""Sampling utilities shared by both plain generation and speculative decoding.
Kept tiny and dependency-free so the probability arithmetic in algorithm.py is easy
to audit against the actual speculative-decoding paper's accept/reject rule.
"""
from __future__ import annotations

import torch


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """logits: [..., vocab]. temperature -> 0 is handled by the caller via greedy
    argmax instead of calling this (dividing by ~0 would blow up numerically)."""
    return torch.softmax(logits / temperature, dim=-1)


def sample_categorical(probs: torch.Tensor, generator: torch.Generator | None = None) -> int:
    """probs: [vocab], a normalized distribution. Returns a single sampled token id."""
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def greedy_token(logits: torch.Tensor) -> int:
    """logits: [vocab]."""
    return int(logits.argmax(dim=-1).item())
