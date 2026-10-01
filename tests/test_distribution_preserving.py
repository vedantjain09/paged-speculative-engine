"""Speculative decoding's correctness guarantee at temperature=0 is
exact-match (tests/test_speculative_single_request.py,
tests/test_scheduler.py). At temperature>0 it's a *weaker but still
precise* guarantee: the rejection-sampling rule (Leviathan et al. 2023 /
Chen et al. 2023) only claims the sampled token's *distribution* exactly
matches the target model's own distribution -- not that any single draw
matches a particular baseline draw, since both are genuinely random.

Testing that claim needs a different tool than exact-match: run many
independent trials, build the empirical distribution of what actually
got sampled, and run a chi-squared goodness-of-fit test against the
exact distribution the target model computes for that position (no
sampling involved on that side -- softmax over its own logits is exact).

Scope: only the first output token after a prefilled prompt is tested.
That token's marginal distribution depends only on target_next_logits at
the prompt's last position (computed once, exactly, during prefill) and
this one step's accept/residual-sample decision -- it doesn't depend on
any earlier speculative accept/reject outcome the way a second or third
output token would, which is what makes it the cleanest token position to
check this specific claim against.
"""
from __future__ import annotations

import torch
from scipy.stats import chisquare

from engine.model import ModelConfig, TinyGPT
from engine.paged_cache import PagedKVCache
from engine.scheduler import Request, run_continuous_speculative_batches
from engine.speculative import batched_speculative_step, prefill_request

VOCAB_SIZE = 12
N_TRIALS = 4000
ALPHA = 0.01  # reject the "distribution matches" null only below this p-value


def _make_models(seed: int) -> tuple[TinyGPT, TinyGPT, ModelConfig]:
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=8, n_layers=1, n_heads=2, d_ff=16, max_seq_len=32)
    draft = TinyGPT(cfg)
    draft.eval()
    target = TinyGPT(cfg)
    target.eval()
    return draft, target, cfg


def _true_first_token_distribution(target: TinyGPT, prompt: list[int]) -> torch.Tensor:
    with torch.no_grad():
        logits, _ = target(torch.tensor([prompt]))
    return torch.softmax(logits[0, -1, :], dim=-1)


def _chi_squared_pvalue(samples: list[int], true_dist: torch.Tensor, vocab_size: int) -> float:
    observed = torch.zeros(vocab_size, dtype=torch.float64)
    for s in samples:
        observed[s] += 1
    # true_dist is a float32 softmax output, so it sums to 1.0 only up to
    # ~1e-7 rounding error -- fine on its own, but scipy.chisquare demands
    # sum(observed) == sum(expected) to a much tighter tolerance than that
    # once both are scaled up by len(samples) (a real, if purely numerical,
    # finding from the first run of this test). Renormalizing expected to
    # sum to exactly len(samples) in float64 is the standard fix and
    # doesn't change what's being tested -- the *shape* of the expected
    # distribution is untouched, only its total is forced exact.
    expected = true_dist.double() * len(samples)
    expected = expected / expected.sum() * len(samples)
    assert expected.min().item() > 3, "expected bin too small for a valid chi-squared test -- adjust N_TRIALS/VOCAB_SIZE"
    stat, p = chisquare(observed.tolist(), expected.tolist())
    return p


def test_single_request_first_token_matches_target_distribution():
    draft, target, cfg = _make_models(seed=10)
    prompt = [1, 2, 3]
    true_dist = _true_first_token_distribution(target, prompt)

    samples = []
    for trial in range(N_TRIALS):
        gen = torch.Generator().manual_seed(trial)
        head_dim = cfg.d_model // cfg.n_heads
        draft_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size=4, num_blocks=16)
        target_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size=4, num_blocks=16)
        draft_next, target_next = prefill_request(draft, target, draft_cache, target_cache, "r", prompt)
        step = batched_speculative_step(
            draft, target, ["r"], draft_cache, target_cache,
            {"r": draft_next}, {"r": target_next}, k=3, temperature=1.0, generator=gen,
        )
        samples.append(step.accepted_tokens["r"][0])

    p = _chi_squared_pvalue(samples, true_dist, VOCAB_SIZE)
    assert p > ALPHA, f"chi-squared p={p:.4f} -- sampled distribution diverges from target's own distribution"


def test_batched_request_first_token_still_matches_target_distribution_despite_sharing_a_step():
    """The stronger version of the same claim: run the request of interest
    inside the multi-request continuous-batching scheduler, sharing every
    batched step with other, differently-prompted requests -- if batch
    sharing introduced any cross-request leakage into the sampling math
    (as opposed to just the KV cache, already checked by the exact-match
    property test), it would bias this distribution even though the
    greedy exact-match tests would never catch it (they only check the
    mode, not the shape of the distribution)."""
    draft, target, cfg = _make_models(seed=11)
    target_prompt = [2, 4, 1]
    true_dist = _true_first_token_distribution(target, target_prompt)

    samples = []
    for trial in range(N_TRIALS):
        gen = torch.Generator().manual_seed(trial)
        head_dim = cfg.d_model // cfg.n_heads
        draft_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size=3, num_blocks=32)
        target_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size=3, num_blocks=32)
        requests = [
            Request("target_req", target_prompt, max_new_tokens=1),
            Request("filler_a", [5, 6], max_new_tokens=3),
            Request("filler_b", [7, 8, 9], max_new_tokens=2),
        ]
        results = run_continuous_speculative_batches(
            draft, target, requests, draft_cache, target_cache,
            max_batch_size=3, k=3, temperature=1.0, generator=gen,
        )
        by_id = {r.request_id: r.output_ids for r in results}
        samples.append(by_id["target_req"][len(target_prompt)])

    p = _chi_squared_pvalue(samples, true_dist, VOCAB_SIZE)
    assert p > ALPHA, f"chi-squared p={p:.4f} -- batch-sharing biased the sampled distribution"
