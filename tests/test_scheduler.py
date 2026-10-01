"""Unit tests for the continuous-batching speculative scheduler: admission
timing, eviction/cache-freeing, and the core correctness claim -- every
request's output exactly matches what it would have produced running
alone, regardless of who else shares its batched steps."""
from __future__ import annotations

import torch

from engine.generate import generate_cached
from engine.model import ModelConfig, TinyGPT
from engine.paged_cache import PagedKVCache
from engine.scheduler import Request, run_continuous_speculative_batches


def _make_models(seed: int, vocab_size: int = 48, d_model: int = 16, n_layers: int = 2, n_heads: int = 2):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=32, max_seq_len=128)
    draft = TinyGPT(cfg)
    draft.eval()
    target = TinyGPT(cfg)
    target.eval()
    return draft, target, cfg


def _run(draft, target, cfg, requests, max_batch_size, k=4, block_size=4, num_blocks=128):
    head_dim = cfg.d_model // cfg.n_heads
    draft_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)
    target_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)
    return run_continuous_speculative_batches(draft, target, requests, draft_cache, target_cache, max_batch_size, k=k, temperature=0.0)


def test_single_request_matches_baseline():
    draft, target, cfg = _make_models(seed=0)
    req = Request("a", [1, 2, 3], 7)
    results = _run(draft, target, cfg, [req], max_batch_size=4)
    baseline = generate_cached(target, torch.tensor([req.prompt_ids]), max_new_tokens=7)
    assert results[0].output_ids == baseline


def test_four_requests_forced_through_small_batch_all_match_solo_baselines():
    draft, target, cfg = _make_models(seed=7)
    requests = [
        Request("a", [1, 2, 3], 8),
        Request("b", [4, 5], 6),
        Request("c", [6, 7, 8, 9], 10),
        Request("d", [2, 2], 5),
    ]
    results = _run(draft, target, cfg, requests, max_batch_size=2, k=4)
    by_id = {r.request_id: r.output_ids for r in results}
    assert len(results) == 4

    for req in requests:
        baseline = generate_cached(target, torch.tensor([req.prompt_ids]), max_new_tokens=req.max_new_tokens)
        assert by_id[req.request_id] == baseline, f"mismatch for {req.request_id}"


def test_all_requests_in_queue_eventually_complete():
    draft, target, cfg = _make_models(seed=3)
    requests = [Request(f"r{i}", [i % 10 + 1, (i + 1) % 10 + 1], 4) for i in range(9)]
    results = _run(draft, target, cfg, requests, max_batch_size=3, k=3)
    assert {r.request_id for r in results} == {f"r{i}" for i in range(9)}


def test_eviction_frees_cache_blocks_back_to_pool():
    draft, target, cfg = _make_models(seed=1)
    head_dim = cfg.d_model // cfg.n_heads
    draft_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size=4, num_blocks=16)
    target_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size=4, num_blocks=16)
    requests = [Request("a", [1, 2], 5)]
    run_continuous_speculative_batches(draft, target, requests, draft_cache, target_cache, max_batch_size=2, k=3)
    # the only request finished and was evicted -- every block it held
    # should be back in the free pool, not leaked.
    assert draft_cache.allocator.num_free == 16
    assert target_cache.allocator.num_free == 16


def test_max_new_tokens_one_for_every_request_does_not_overshoot():
    draft, target, cfg = _make_models(seed=2)
    requests = [Request("a", [1, 1], 1), Request("b", [2, 2], 1)]
    results = _run(draft, target, cfg, requests, max_batch_size=4, k=4)
    for r in results:
        prompt_len = 2
        assert len(r.output_ids) - prompt_len == 1


def test_batch_pressure_does_not_affect_output_across_many_seeds():
    # the real stress case: force 6 requests through a batch size of 2
    # (guaranteeing multiple rounds of admission/eviction mid-generation)
    # across several seeds/configs, and require every single one to match
    # its own solo greedy baseline exactly.
    for seed in range(4):
        draft, target, cfg = _make_models(seed=seed, vocab_size=40, d_model=12)
        requests = [Request(f"r{i}", [(i + j) % 10 + 1 for j in range(2 + i % 3)], 5 + i % 4) for i in range(6)]
        results = _run(draft, target, cfg, requests, max_batch_size=2, k=3)
        by_id = {r.request_id: r.output_ids for r in results}
        for req in requests:
            baseline = generate_cached(target, torch.tensor([req.prompt_ids]), max_new_tokens=req.max_new_tokens)
            assert by_id[req.request_id] == baseline, f"seed={seed} mismatch for {req.request_id}"
