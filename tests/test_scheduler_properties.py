"""The property test this whole project exists to pass: for a randomly
generated set of requests (random prompts, random max_new_tokens), run
through the continuous-batching speculative scheduler with a random
max_batch_size and random k/block_size -- forcing arbitrary, data-dependent
admission timing and per-request accept-length heterogeneity within the
same shared paged pool -- every single request's final output must
exactly match what it would have produced running completely alone.

This is the direct generalization of inference-batching-lab's own
property test (every batched strategy must byte-match run_sequential) to
the much harder case this project is actually about: speculative
decoding's variable per-request commit lengths, batched through a shared
paged cache. If paged_cache.py had any aliasing bug, or
batched_forward.py's key_positions/slicing logic had any off-by-one, this
is the test most likely to catch it -- a bug there would show up as one
request's tokens silently drifting once it shares a step with a neighbor
accepting a different number of draft tokens, which only happens with
genuine multi-request heterogeneity under batch pressure.
"""
from __future__ import annotations

import torch
from hypothesis import HealthCheck, given, settings, strategies as st

from engine.generate import generate_cached
from engine.model import ModelConfig, TinyGPT
from engine.paged_cache import PagedKVCache
from engine.scheduler import Request, run_continuous_speculative_batches

VOCAB_SIZE = 32


def _make_models(seed: int) -> tuple[TinyGPT, TinyGPT, ModelConfig]:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=12, n_layers=2, n_heads=2, d_ff=24, max_seq_len=256)
    draft = TinyGPT(cfg)
    draft.eval()
    target = TinyGPT(cfg)
    target.eval()
    return draft, target, cfg


request_strategy = st.tuples(
    st.lists(st.integers(min_value=1, max_value=VOCAB_SIZE - 1), min_size=1, max_size=5),  # prompt
    st.integers(min_value=1, max_value=8),  # max_new_tokens
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(min_value=0, max_value=1000),
    requests_spec=st.lists(request_strategy, min_size=1, max_size=6, unique_by=lambda r: tuple(r[0])),
    max_batch_size=st.integers(min_value=1, max_value=4),
    k=st.integers(min_value=1, max_value=5),
    block_size=st.integers(min_value=1, max_value=5),
)
def test_batched_output_always_matches_solo_baseline(seed, requests_spec, max_batch_size, k, block_size):
    draft, target, cfg = _make_models(seed)
    requests = [Request(f"r{i}", prompt, max_new) for i, (prompt, max_new) in enumerate(requests_spec)]

    head_dim = cfg.d_model // cfg.n_heads
    # generous pool: worst case every request's full length resident at once
    total_tokens_ceiling = sum(len(r.prompt_ids) + r.max_new_tokens + k + 1 for r in requests)
    num_blocks = (total_tokens_ceiling // block_size + 1) * len(requests) + 4

    draft_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)
    target_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)

    results = run_continuous_speculative_batches(
        draft, target, requests, draft_cache, target_cache, max_batch_size, k=k, temperature=0.0,
    )
    by_id = {r.request_id: r.output_ids for r in results}

    assert set(by_id.keys()) == {r.request_id for r in requests}

    for req in requests:
        baseline = generate_cached(target, torch.tensor([req.prompt_ids]), max_new_tokens=req.max_new_tokens)
        assert by_id[req.request_id] == baseline, (
            f"seed={seed} k={k} block_size={block_size} max_batch_size={max_batch_size} "
            f"request={req.request_id} prompt={req.prompt_ids} max_new={req.max_new_tokens}"
        )
