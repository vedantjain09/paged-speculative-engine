"""Real measured throughput, three configurations:

1. Plain continuous batching through the paged cache, no speculation at
   all (one model, one token per request per step) -- the baseline every
   other number here is measured against.
2. Speculative decoding + continuous batching + paging, with an
   INDEPENDENT, untrained draft model -- the honest floor. These are
   random-weight models (see README "Honest limitations": no network
   access to download a real pretrained checkpoint in this sandbox), so
   two independently-initialized models agree at roughly the 1/vocab_size
   floor -- this measures the real overhead of running two full models
   plus the accept/reject machinery when the draft essentially never
   helps, which is itself a real and useful number (it answers "how bad
   is the downside if your draft model is badly misaligned").
3. Speculative decoding + continuous batching + paging, with the draft
   model set to literally be the target model (self-speculation, 100%
   acceptance every round) -- the measured ceiling: every draft token is
   accepted, so this shows the maximum throughput the scheduling/paging
   machinery itself can deliver, independent of how good any particular
   draft model is.

Both (2) and (3) are standard benchmarking configurations in the actual
speculative-decoding literature (an identical/self-draft run isolates the
mechanism's overhead from any particular draft model's quality), not
something invented to make a number look better -- see README for why
reporting both instead of only the flattering one matters here.
"""
from __future__ import annotations

import time

import torch

from engine.batched_forward import batched_forward_append
from engine.model import ModelConfig, TinyGPT
from engine.paged_cache import PagedKVCache
from engine.scheduler import Request, RequestResult, run_continuous_speculative_batches


@torch.no_grad()
def run_plain_continuous_batches(
    model: TinyGPT, requests: list[Request], cache: PagedKVCache, max_batch_size: int,
) -> list[RequestResult]:
    """No speculation: one token per request per batched step. Mirrors
    inference-batching-lab's run_continuous_batches, but through the
    paged cache instead of per-step padding."""
    queue = list(requests)
    active: dict[str, dict] = {}
    results: list[RequestResult] = []
    t_start = time.perf_counter()

    def admit() -> None:
        while queue and len(active) < max_batch_size:
            req = queue.pop(0)
            cache.create_request(req.request_id)
            prompt = torch.tensor([req.prompt_ids])
            logits = batched_forward_append(model, cache, [req.request_id], prompt)
            next_token = int(logits[0, -1, :].argmax().item())
            active[req.request_id] = {"req": req, "tokens": [*req.prompt_ids, next_token], "generated": 1}

    admit()
    while active or queue:
        if not active:
            admit()
            continue
        ids = list(active.keys())
        next_inputs = torch.tensor([[active[rid]["tokens"][-1]] for rid in ids])
        logits = batched_forward_append(model, cache, ids, next_inputs)
        finished = []
        for i, rid in enumerate(ids):
            a = active[rid]
            next_token = int(logits[i, -1, :].argmax().item())
            a["tokens"].append(next_token)
            a["generated"] += 1
            if a["generated"] >= a["req"].max_new_tokens:
                finished.append(rid)
        t_now = time.perf_counter()
        for rid in finished:
            a = active.pop(rid)
            results.append(RequestResult(rid, a["tokens"], t_start, t_now))
            cache.free_request(rid)
        admit()

    return results


def _make_workload(n_requests: int, prompt_len: int, max_new_tokens: int, vocab_size: int) -> list[Request]:
    return [
        Request(f"r{i}", [((i * 7 + j) % (vocab_size - 1)) + 1 for j in range(prompt_len)], max_new_tokens)
        for i in range(n_requests)
    ]


def benchmark_config(name: str, draft: TinyGPT, target: TinyGPT, cfg: ModelConfig, requests: list[Request], max_batch_size: int, k: int, block_size: int) -> dict:
    head_dim = cfg.d_model // cfg.n_heads
    total_tokens_ceiling = sum(len(r.prompt_ids) + r.max_new_tokens + k + 2 for r in requests)
    num_blocks = total_tokens_ceiling // block_size + len(requests) * 4

    draft_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)
    target_cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)

    t0 = time.perf_counter()
    results = run_continuous_speculative_batches(draft, target, requests, draft_cache, target_cache, max_batch_size, k=k, temperature=0.0)
    wall = time.perf_counter() - t0

    total_new_tokens = sum(len(r.output_ids) - len(req.prompt_ids) for r, req in zip(results, sorted(requests, key=lambda r: r.request_id)))
    return {"name": name, "wall_s": wall, "throughput_tok_s": total_new_tokens / wall, "total_new_tokens": total_new_tokens}


def benchmark_plain(name: str, target: TinyGPT, cfg: ModelConfig, requests: list[Request], max_batch_size: int, block_size: int) -> dict:
    head_dim = cfg.d_model // cfg.n_heads
    total_tokens_ceiling = sum(len(r.prompt_ids) + r.max_new_tokens + 2 for r in requests)
    num_blocks = total_tokens_ceiling // block_size + len(requests) * 4
    cache = PagedKVCache(cfg.n_layers, cfg.n_heads, head_dim, block_size, num_blocks)

    t0 = time.perf_counter()
    results = run_plain_continuous_batches(target, requests, cache, max_batch_size)
    wall = time.perf_counter() - t0

    by_id = {r.request_id: r for r in results}
    total_new_tokens = sum(len(by_id[req.request_id].output_ids) - len(req.prompt_ids) for req in requests)
    return {"name": name, "wall_s": wall, "throughput_tok_s": total_new_tokens / wall, "total_new_tokens": total_new_tokens}


if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=4, n_heads=4, d_ff=256, max_seq_len=256)
    target_model = TinyGPT(cfg)
    target_model.eval()
    draft_model_independent = TinyGPT(cfg)  # fresh, independent random weights -- the floor
    draft_model_independent.eval()

    N_REQUESTS, PROMPT_LEN, MAX_NEW, MAX_BATCH, K, BLOCK = 24, 6, 20, 6, 4, 8
    requests_plain = _make_workload(N_REQUESTS, PROMPT_LEN, MAX_NEW, cfg.vocab_size)
    requests_floor = _make_workload(N_REQUESTS, PROMPT_LEN, MAX_NEW, cfg.vocab_size)
    requests_ceiling = _make_workload(N_REQUESTS, PROMPT_LEN, MAX_NEW, cfg.vocab_size)

    plain = benchmark_plain("plain continuous batching (no speculation)", target_model, cfg, requests_plain, MAX_BATCH, BLOCK)
    floor = benchmark_config("speculative + batching + paging, independent draft (FLOOR)", draft_model_independent, target_model, cfg, requests_floor, MAX_BATCH, K, BLOCK)
    ceiling = benchmark_config("speculative + batching + paging, draft==target (CEILING)", target_model, target_model, cfg, requests_ceiling, MAX_BATCH, K, BLOCK)

    print(f"workload: {N_REQUESTS} requests, prompt_len={PROMPT_LEN}, max_new_tokens={MAX_NEW}, max_batch_size={MAX_BATCH}, k={K}, block_size={BLOCK}")
    print()
    for r in (plain, floor, ceiling):
        print(f"{r['name']:60s} wall={r['wall_s']:.3f}s  throughput={r['throughput_tok_s']:.1f} tok/s  total_tokens={r['total_new_tokens']}")
    print()
    print(f"floor/plain speedup:   {floor['throughput_tok_s'] / plain['throughput_tok_s']:.3f}x")
    print(f"ceiling/plain speedup: {ceiling['throughput_tok_s'] / plain['throughput_tok_s']:.3f}x")
