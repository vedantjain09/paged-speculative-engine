"""Correctness proof for the paged-cache-backed speculative decoding path,
single request: at temperature=0 (greedy), speculative decoding is
provably exact-match with plain target-only generation -- the draft model
only ever proposes, the target model's own argmax is always what actually
gets accepted or substituted. This is the same guarantee
speculative-decoding-lab's test_algorithm.py checks against a
contiguous-tensor cache; here it's checked against the paged cache and the
batch-shaped (batch=1) step function instead."""
from __future__ import annotations

import torch

from engine.generate import generate_cached, speculative_generate_batched
from engine.model import ModelConfig, TinyGPT
from engine.paged_cache import PagedKVCache


def _make_models(seed: int, vocab_size: int = 64, d_model: int = 16, n_layers: int = 2, n_heads: int = 2):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=32, max_seq_len=128)
    draft = TinyGPT(cfg)
    draft.eval()
    target = TinyGPT(cfg)
    target.eval()
    return draft, target, cfg


def _run_speculative(draft, target, cfg, prompt, max_new_tokens, k, block_size=4, num_blocks=64):
    head_dim = cfg.d_model // cfg.n_heads
    draft_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)
    target_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)
    return speculative_generate_batched(draft, target, draft_cache, target_cache, "r1", prompt, max_new_tokens, k=k, temperature=0.0)


def test_exact_match_against_sequential_baseline():
    draft, target, cfg = _make_models(seed=0)
    prompt = [1, 2, 3, 4]
    baseline = generate_cached(target, torch.tensor([prompt]), max_new_tokens=10)
    spec = _run_speculative(draft, target, cfg, prompt, max_new_tokens=10, k=4)
    assert spec == baseline


def test_exact_match_holds_across_several_seeds_and_configs():
    for seed in range(5):
        draft, target, cfg = _make_models(seed=seed, vocab_size=48, d_model=12, n_layers=2, n_heads=2)
        prompt = [seed % 10 + 1, (seed + 1) % 10 + 1, (seed + 2) % 10 + 1]
        baseline = generate_cached(target, torch.tensor([prompt]), max_new_tokens=8)
        spec = _run_speculative(draft, target, cfg, prompt, max_new_tokens=8, k=3)
        assert spec == baseline, f"mismatch at seed={seed}"


def test_exact_match_when_prompt_longer_than_block_size():
    # block_size=4, prompt length 9 -- prefill alone spans 3 blocks, so
    # this exercises append() crossing block boundaries during prefill,
    # not just during decode.
    draft, target, cfg = _make_models(seed=1)
    prompt = list(range(1, 10))
    baseline = generate_cached(target, torch.tensor([prompt]), max_new_tokens=6)
    spec = _run_speculative(draft, target, cfg, prompt, max_new_tokens=6, k=4, block_size=4)
    assert spec == baseline


def test_exact_match_when_k_does_not_divide_block_size():
    # k=3 against block_size=4: draft/verify commits don't land on block
    # boundaries, so every step's truncate() lands mid-block somewhere.
    draft, target, cfg = _make_models(seed=2)
    prompt = [5, 6, 7]
    baseline = generate_cached(target, torch.tensor([prompt]), max_new_tokens=11)
    spec = _run_speculative(draft, target, cfg, prompt, max_new_tokens=11, k=3, block_size=4)
    assert spec == baseline


def test_exact_match_when_k_larger_than_block_size():
    # k=6 > block_size=4: a single verification call's worth of draft
    # tokens alone spans more than one block.
    draft, target, cfg = _make_models(seed=3)
    prompt = [2, 4, 6]
    baseline = generate_cached(target, torch.tensor([prompt]), max_new_tokens=13)
    spec = _run_speculative(draft, target, cfg, prompt, max_new_tokens=13, k=6, block_size=4)
    assert spec == baseline


def test_max_new_tokens_one_does_not_overshoot():
    draft, target, cfg = _make_models(seed=4)
    prompt = [1, 1, 1]
    spec = _run_speculative(draft, target, cfg, prompt, max_new_tokens=1, k=4)
    assert len(spec) - len(prompt) == 1


def test_draft_equal_to_target_still_exact_match():
    # Self-speculation (draft IS target): every draft token should be
    # accepted every round, so the whole generation collapses to k+1
    # tokens per speculative step -- still must match target-only greedy
    # byte-for-byte, since the correctness guarantee doesn't depend on how
    # good or bad the draft model is.
    _, target, cfg = _make_models(seed=5)
    prompt = [3, 3, 3]
    baseline = generate_cached(target, torch.tensor([prompt]), max_new_tokens=9)
    spec = _run_speculative(target, target, cfg, prompt, max_new_tokens=9, k=4)
    assert spec == baseline
