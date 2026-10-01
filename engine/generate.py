"""Plain sequential, non-speculative, non-batched generation -- the
correctness oracle every other path in this repo is checked against.
Reused unmodified from inference-batching-lab/serving/generate.py except
for the import path (serving.model -> engine.model)."""
from __future__ import annotations

import torch

from engine.model import TinyGPT


@torch.no_grad()
def generate_naive(model: TinyGPT, prompt_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    model.eval()
    tokens = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits, _ = model(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens[0].tolist()


@torch.no_grad()
def generate_cached(model: TinyGPT, prompt_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    model.eval()
    tokens = prompt_ids.clone()

    logits, past_kv = model(tokens)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=1)

    for _ in range(max_new_tokens - 1):
        seq_len_so_far = tokens.shape[1] - 1
        logits, past_kv = model(next_token, past_kv=past_kv, position_offset=seq_len_so_far)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens[0].tolist()


@torch.no_grad()
def speculative_generate_batched(
    draft_model: TinyGPT,
    target_model: TinyGPT,
    draft_cache,
    target_cache,
    request_id: str,
    prompt_ids: list[int],
    max_new_tokens: int,
    k: int = 4,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Single-request driver over the batch-shaped machinery in
    speculative.py (called here with a request_ids list of length 1) --
    this is the minimal case that has to be correct before Section 4 adds
    real multi-request heterogeneity on top of the same primitives."""
    from engine.speculative import batched_speculative_step, prefill_request

    draft_next, target_next = prefill_request(
        draft_model, target_model, draft_cache, target_cache, request_id, prompt_ids
    )
    tokens = list(prompt_ids)

    while len(tokens) - len(prompt_ids) < max_new_tokens:
        result = batched_speculative_step(
            draft_model, target_model, [request_id], draft_cache, target_cache,
            {request_id: draft_next}, {request_id: target_next}, k, temperature, generator,
        )
        tokens.extend(result.accepted_tokens[request_id])
        draft_next = result.draft_next_logits[request_id]
        target_next = result.target_next_logits[request_id]

    overshoot = (len(tokens) - len(prompt_ids)) - max_new_tokens
    return tokens[: len(tokens) - overshoot] if overshoot > 0 else tokens
