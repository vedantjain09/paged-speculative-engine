"""Continuous-batching scheduler driving multi-request speculative
decoding -- the integration layer where this whole project's premise
actually gets tested. Every request in `active` goes through the SAME
batched_speculative_step() call each round, but each one independently
accepts a different number of draft tokens (0..k, data-dependent on how
well its own draft/target disagree), and each one's cache gets truncated
back to its own accepted length, through the same shared paged pool,
without ever touching another request's blocks. If there were any
aliasing bug in paged_cache.py or any mis-indexing in batched_forward.py,
this is where it would show up: a request's output would start drifting
away from what it would have produced running alone, as soon as it
shares a batched step with neighbors that accept a different number of
tokens than it does.

Admission/eviction policy mirrors inference-batching-lab's
run_continuous_batches: a finished request's slot is backfilled from the
queue immediately, not held empty until every other active request also
finishes (that would be static batching's head-of-line-blocking cost).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from engine.model import TinyGPT
from engine.paged_cache import PagedKVCache
from engine.speculative import batched_speculative_step, prefill_request


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int


@dataclass
class RequestResult:
    request_id: str
    output_ids: list[int]
    submitted_at: float
    completed_at: float

    @property
    def latency_s(self) -> float:
        return self.completed_at - self.submitted_at


@dataclass
class _ActiveRequest:
    request: Request
    tokens: list[int]
    draft_next_logits: torch.Tensor
    target_next_logits: torch.Tensor
    generated: int = 0
    submitted_at: float = field(default_factory=time.perf_counter)


@torch.no_grad()
def run_continuous_speculative_batches(
    draft_model: TinyGPT,
    target_model: TinyGPT,
    requests: list[Request],
    draft_cache: PagedKVCache,
    target_cache: PagedKVCache,
    max_batch_size: int,
    k: int = 4,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> list[RequestResult]:
    queue = list(requests)
    active: dict[str, _ActiveRequest] = {}
    results: list[RequestResult] = []
    t_start = time.perf_counter()

    def admit_new_requests() -> None:
        while queue and len(active) < max_batch_size:
            req = queue.pop(0)
            draft_next, target_next = prefill_request(
                draft_model, target_model, draft_cache, target_cache, req.request_id, req.prompt_ids
            )
            active[req.request_id] = _ActiveRequest(
                request=req,
                tokens=list(req.prompt_ids),
                draft_next_logits=draft_next,
                target_next_logits=target_next,
                submitted_at=t_start,
            )

    admit_new_requests()
    while active or queue:
        if not active:
            admit_new_requests()
            continue

        ids = list(active.keys())
        draft_next_map = {rid: active[rid].draft_next_logits for rid in ids}
        target_next_map = {rid: active[rid].target_next_logits for rid in ids}

        step = batched_speculative_step(
            draft_model, target_model, ids, draft_cache, target_cache,
            draft_next_map, target_next_map, k, temperature, generator,
        )

        finished: list[str] = []
        for rid in ids:
            a = active[rid]
            a.tokens.extend(step.accepted_tokens[rid])
            a.generated += len(step.accepted_tokens[rid])
            a.draft_next_logits = step.draft_next_logits[rid]
            a.target_next_logits = step.target_next_logits[rid]
            if a.generated >= a.request.max_new_tokens:
                finished.append(rid)

        t_now = time.perf_counter()
        for rid in finished:
            a = active.pop(rid)
            overshoot = a.generated - a.request.max_new_tokens
            final_tokens = a.tokens[: len(a.tokens) - overshoot] if overshoot > 0 else a.tokens
            results.append(RequestResult(rid, final_tokens, a.submitted_at, t_now))
            draft_cache.free_request(rid)
            target_cache.free_request(rid)

        admit_new_requests()  # backfill freed slots immediately

    return results
