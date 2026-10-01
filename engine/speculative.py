"""Batch-aware speculative decoding on top of the paged KV-cache.

The accept/reject/resample math is the exact, unmodified rejection-sampling
rule from speculative-decoding-lab's algorithm.py (provably
distribution-preserving -- see that repo's docstring for the derivation),
reused here via sampling.py. What's new in this file is how KV state
moves: every forward call through batched_forward_append() appends a
UNIFORM number of new tokens for every request in the batch -- that's
unavoidable, one tensor shape can't vary per row. Drafting therefore
proceeds in k rounds of width-1 calls; verification is one width-k call.

The part that makes this project's whole premise true: after
verification, each request's accepted length (0..k) is data-dependent and
different per request. That heterogeneity is handled *entirely* by each
request's own cache.truncate() call afterward -- trivial with a paged,
per-request block table (it just frees trailing blocks), and genuinely
awkward with a contiguous padded tensor (you'd either waste memory
keeping rejected positions allocated, or pay to shift every row's data by
a different, position-dependent amount to close the gap). This file never
has to reconcile different per-row tensor widths inside a forward call --
only between them, via paged_cache.py's accounting.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from engine.batched_forward import batched_forward_append
from engine.model import TinyGPT
from engine.paged_cache import PagedKVCache
from engine.sampling import greedy_token, sample_categorical, softmax_with_temperature


@dataclass
class BatchStepResult:
    accepted_tokens: dict[str, list[int]]
    num_draft_accepted: dict[str, int]
    draft_next_logits: dict[str, torch.Tensor]
    target_next_logits: dict[str, torch.Tensor]


def _residual_sample(p: torch.Tensor, q: torch.Tensor, generator) -> int:
    """normalize(max(0, p - q)) -- the exact correction term that makes
    rejection sampling distribution-preserving (Leviathan et al. 2023 /
    Chen et al. 2023); see speculative-decoding-lab's algorithm.py."""
    residual = torch.clamp(p - q, min=0.0)
    total = residual.sum()
    residual = p if total <= 0 else residual / total
    return sample_categorical(residual, generator)


@torch.no_grad()
def prefill_request(
    draft_model: TinyGPT,
    target_model: TinyGPT,
    draft_cache: PagedKVCache,
    target_cache: PagedKVCache,
    request_id: str,
    prompt_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeds both caches with the prompt (each request prefilled on its
    own -- different requests generally have different prompt lengths, so
    this one call is never batched across requests, same as every
    continuous-batching scheduler's admission step). Returns the first
    next-token-prediction logits for both models, the "byproduct" that
    the first speculative step's draft/target_next_logits comes from."""
    draft_cache.create_request(request_id)
    target_cache.create_request(request_id)
    prompt = torch.tensor([prompt_ids])
    draft_logits = batched_forward_append(draft_model, draft_cache, [request_id], prompt)
    target_logits = batched_forward_append(target_model, target_cache, [request_id], prompt)
    return draft_logits[0, -1, :], target_logits[0, -1, :]


@torch.no_grad()
def batched_speculative_step(
    draft_model: TinyGPT,
    target_model: TinyGPT,
    request_ids: list[str],
    draft_cache: PagedKVCache,
    target_cache: PagedKVCache,
    draft_next_logits: dict[str, torch.Tensor],
    target_next_logits: dict[str, torch.Tensor],
    k: int,
    temperature: float,
    generator: torch.Generator | None = None,
) -> BatchStepResult:
    base_len_draft = {rid: draft_cache.length(rid) for rid in request_ids}
    base_len_target = {rid: target_cache.length(rid) for rid in request_ids}

    # --- draft phase: k uniform-width-1 batched rounds ---
    pending = dict(draft_next_logits)
    draft_tokens: dict[str, list[int]] = {rid: [] for rid in request_ids}
    draft_probs: dict[str, list[torch.Tensor]] = {rid: [] for rid in request_ids}

    for _ in range(k):
        chosen: dict[str, int] = {}
        for rid in request_ids:
            if temperature > 0:
                dist = softmax_with_temperature(pending[rid], temperature)
                tok = sample_categorical(dist, generator)
            else:
                tok = greedy_token(pending[rid])
                dist = torch.nn.functional.one_hot(torch.tensor(tok), pending[rid].shape[-1]).float()
            chosen[rid] = tok
            draft_tokens[rid].append(tok)
            draft_probs[rid].append(dist)

        input_ids = torch.tensor([[chosen[rid]] for rid in request_ids])
        logits = batched_forward_append(draft_model, draft_cache, request_ids, input_ids)
        for i, rid in enumerate(request_ids):
            pending[rid] = logits[i, -1, :]

    # --- target verification: ONE uniform-width-k batched call ---
    draft_token_tensor = torch.tensor([draft_tokens[rid] for rid in request_ids])  # [batch, k]
    target_logits = batched_forward_append(target_model, target_cache, request_ids, draft_token_tensor)

    def to_dist(row: torch.Tensor) -> torch.Tensor:
        return softmax_with_temperature(row, temperature) if temperature > 0 else row

    accepted: dict[str, list[int]] = {}
    num_accepted: dict[str, int] = {}

    for i, rid in enumerate(request_ids):
        # dists[0] verifies draft_tokens[rid][0] against the already-known
        # target_next_logits[rid] (a byproduct of the previous step's bonus
        # call, never recomputed); dists[1..k] come from this call.
        dists = [to_dist(target_next_logits[rid])] + [to_dist(target_logits[i, j, :]) for j in range(k)]
        acc: list[int] = []
        n_acc = 0
        if temperature > 0:
            for j in range(k):
                p_j, q_j, x_j = dists[j], draft_probs[rid][j], draft_tokens[rid][j]
                accept_prob = min(1.0, (p_j[x_j] / q_j[x_j]).item()) if q_j[x_j] > 0 else 0.0
                u = torch.rand((), generator=generator).item()
                if u < accept_prob:
                    acc.append(x_j)
                    n_acc += 1
                else:
                    acc.append(_residual_sample(p_j, q_j, generator))
                    break
            else:
                acc.append(sample_categorical(dists[k], generator))
        else:
            for j in range(k):
                target_argmax = int(dists[j].argmax(dim=-1).item())
                if draft_tokens[rid][j] == target_argmax:
                    acc.append(draft_tokens[rid][j])
                    n_acc += 1
                else:
                    acc.append(target_argmax)
                    break
            else:
                acc.append(int(dists[k].argmax(dim=-1).item()))
        accepted[rid] = acc
        num_accepted[rid] = n_acc

    # --- commit: truncate each request's cache back to its OWN accepted
    # length, independently. This is the step that only works cleanly
    # because the cache is paged -- see module docstring.
    for rid in request_ids:
        draft_cache.truncate(rid, base_len_draft[rid] + num_accepted[rid])
        target_cache.truncate(rid, base_len_target[rid] + num_accepted[rid])

    # --- bonus/corrected-token step: one more uniform-width-1 batched call
    # for both models, extending the (now-truncated) cache by the final
    # accepted/corrected token and producing next round's next_logits as a
    # byproduct -- avoids ever re-feeding a token either cache already saw.
    final_tokens = torch.tensor([[accepted[rid][-1]] for rid in request_ids])
    draft_bonus_logits = batched_forward_append(draft_model, draft_cache, request_ids, final_tokens)
    target_bonus_logits = batched_forward_append(target_model, target_cache, request_ids, final_tokens)

    new_draft_next = {rid: draft_bonus_logits[i, -1, :] for i, rid in enumerate(request_ids)}
    new_target_next = {rid: target_bonus_logits[i, -1, :] for i, rid in enumerate(request_ids)}

    return BatchStepResult(accepted, num_accepted, new_draft_next, new_target_next)
