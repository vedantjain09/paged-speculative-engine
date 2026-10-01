# paged-speculative-engine

A from-scratch LLM serving engine that combines the three load-bearing
optimizations of production inference systems -- continuous batching, a
paged KV-cache, and speculative decoding -- and solves the problem that
shows up only when you actually combine them: **speculative decoding
breaks continuous batching's core assumption.**

## Why this project

Each of these three techniques is well covered in isolation, including in
two of my own earlier projects (`inference-batching-lab`: continuous
batching with padding; `speculative-decoding-lab`: draft/target rejection
sampling). What almost nothing covers -- including most tutorials and
blog posts -- is that they don't compose for free.

Continuous batching, as normally implemented, assumes every request in
the batch advances by exactly one token per decode step. That's true for
plain autoregressive decoding. It's false the moment speculative decoding
is involved: one request's draft tokens might all be accepted this step
(it advances by K+1 tokens), another's might all be rejected (it advances
by 1), and a third might accept some prefix in between -- all in the same
batched step, against the same shared cache pool. A KV-cache
implementation built around "every row grows by 1" (the padding-based
approach in `inference-batching-lab`) can't express that. A real paged
allocator -- fixed-size blocks, allocated and freed per request
independently -- can, but only if the commit logic is written to handle
per-row variable-length writes correctly, which is the actual hard part
of this repo.

This is a real, current problem in production inference engines (vLLM
and TensorRT-LLM only converged on clean solutions for this in the last
couple of years), not a toy composition exercise.

## The composition problem, precisely

The actual resolution turned out to be narrower than it first looks, and
it's worth stating precisely because the obvious framing ("variable-width
batched tensor op") is wrong:

- **The forward pass itself is always uniform-width across the batch.**
  Tensor shapes don't bend: drafting is `k` sequential rounds of a
  width-1 batched call (every active request proposes one more draft
  token per round), and verification is exactly one width-`k` batched
  call (every active request gets its `k` draft tokens checked at once).
  There is no point where the model is asked to process a different
  number of tokens for different rows in the same call.
- **The heterogeneity is entirely between calls, not within one.** After
  verification, request A might have accepted all `k` tokens and request
  B might have accepted zero -- but that's resolved by each request
  independently truncating its own KV-cache back to its own accepted
  length (`cache.truncate(request_id, accepted_len)`), not by any op
  having to process rows of different widths at once.
- **This is exactly where paging earns its cost.** With a padded,
  contiguous tensor (the `inference-batching-lab` approach), truncating
  one row to a shorter length either wastes the padding or requires
  shifting every other row's data to repack the tensor. With paging,
  truncation is just freeing that request's trailing blocks back to the
  pool -- an O(blocks freed) free-list operation that touches nothing
  belonging to any other request.

So the real engineering problem this repo solves isn't "make a tensor op
handle variable widths." It's "make per-request truncation of a shared
cache, between otherwise-uniform batched calls, correct and cheap" --
and paging is the mechanism that makes it both.

## Architecture

```
engine/
  model.py        -- TinyGPT: hand-written attention + explicit per-layer
                     KV-cache (reused from inference-batching-lab, unmodified)
  sampling.py      -- softmax/categorical-sample/greedy helpers (reused from
                       speculative-decoding-lab, unmodified)
  paged_cache.py   -- BlockAllocator (free-list over a fixed-size block pool)
                      + PagedKVCache (per-request block table, variable-length
                      append, gather-for-attention)
  speculative.py   -- batch-aware draft/verify/accept-reject-resample step,
                      reading/writing through the paged cache instead of a
                      growing contiguous tensor
  scheduler.py     -- continuous-batching admission/eviction loop driving
                      multi-request speculative steps -- the integration
                      layer where the variable-commit-length problem lives
  generate.py      -- plain sequential, non-speculative, non-batched
                      generation -- the correctness oracle everything else
                      is checked against
  benchmark.py     -- real measured throughput, floor and ceiling configs
                      (see "Honest limitations")
tests/             -- unit, exact-match, property-based (Hypothesis), and
                       statistical (chi-squared) correctness tests
```

## Correctness, proven four ways

1. **Exact-match at temperature=0** (`tests/test_speculative_single_request.py`,
   `tests/test_scheduler.py`): at greedy decoding, speculative decoding's
   accept/reject rule is defined to reproduce the target model's own
   greedy output exactly, token for token. Tested for single requests,
   four requests forced through a batch size of 2, eviction/cache-free
   correctness, and six requests across four different random seeds.
2. **Hypothesis property test over the scheduler** (`tests/test_scheduler_properties.py`,
   300 generated examples): random prompts, random `max_new_tokens`,
   random `max_batch_size`, random `k`, random `block_size` -- every
   single request's final output must exactly match what it would have
   produced running completely alone through `generate_cached`. This is
   the test most likely to catch a paged-cache aliasing bug or an
   off-by-one in the batched forward pass's position indexing, because
   it forces genuine multi-request heterogeneity under batch pressure.
3. **Hypothesis property test over the cache allocator** (`tests/test_paged_cache_properties.py`,
   150 examples): random sequences of block allocation/free/append never
   corrupt the free-list or double-allocate a block. One edge case
   (legitimate pool exhaustion under an adversarial append sequence) is
   discarded via `assume(False)` rather than hidden -- see commit history
   for the real `MemoryError` reproduction that motivated it.
4. **Statistical distribution-preserving proof at temperature>0**
   (`tests/test_distribution_preserving.py`): at temperature=0, "correct"
   means exact-match; above zero, the rejection-sampling guarantee is
   that the *distribution* of the sampled token matches the target
   model's own distribution, not that any individual draw matches a
   baseline draw. Tested with 4000 independent trials and a chi-squared
   goodness-of-fit test, both for a single request and for a request
   sharing every batched step with two other, differently-prompted
   requests. Real measured result: chi-squared statistic=8.81, p=0.64
   against the target's exact 12-token distribution (comfortably above
   the alpha=0.01 rejection threshold).

29/29 tests passing (see commit history for the two real bugs this
testing surfaced and fixed: a Hypothesis-discoverable pool-exhaustion
exception, and a scipy floating-point tolerance mismatch).

## Benchmark results (real, measured, CPU)

Three configurations, same workload (24 requests, prompt length 6,
20 new tokens each, max batch size 6, draft length k=4, block size 8),
all running the identical scheduler/paging code path:

| Configuration | Throughput | vs. plain batching |
|---|---|---|
| Plain continuous batching, no speculation (1 model) | 2640 tok/s | 1.00x (baseline) |
| Speculative + batching + paging, independent random draft (**floor**) | 359 tok/s | 0.14x |
| Speculative + batching + paging, draft model == target model (**ceiling**) | 1643 tok/s | 0.62x |

Run it yourself: `python3 -m engine.benchmark`.

**Read these numbers for what they actually show, not what the
speculative-decoding pitch usually implies.** On real GPU-served LLMs,
speculative decoding wins because decode steps are memory-bandwidth
bound, not compute bound -- verifying `k` draft tokens in one forward
pass costs barely more than verifying one, because the bottleneck is
moving weights through memory, not the extra multiply-adds. That's the
mechanism this whole technique is built on.

This repo runs on CPU with intentionally tiny models, where that
assumption does not hold: a width-`k` forward pass costs meaningfully
more than a width-1 one (compute-bound, no memory-bandwidth slack to
hide the extra work in), and every draft round still pays full Python/
PyTorch call overhead regardless of model size. The result is that even
the **ceiling** configuration -- 100% draft acceptance, the best case
speculative decoding can ever produce -- is *slower* than plain batching
here (0.62x), because it still pays `k` sequential draft calls plus one
verify call to produce the same `k` tokens plain decoding gets from `k`
single calls, and nothing here is memory-bandwidth bound enough to make
the extra calls free. The floor configuration (0.14x) adds the honest
worst case on top: an untrained, independent draft model whose tokens
get rejected almost immediately, so most of its compute is pure waste.

This is a real, honestly-reported negative result, not a bug to paper
over: it's exactly the regime (compute-bound, call-overhead-dominated,
no memory-bandwidth bottleneck) where speculative decoding is *not*
supposed to help, and the benchmark shows that precisely. The point of
this repo is the correctness of the composition under paging and batch
heterogeneity (see "Correctness, proven four ways"), not a CPU speedup
claim -- see "Honest limitations" below for why a real speedup
measurement isn't possible here at all.

## Honest limitations

- **No GPU in this environment.** `torch.cuda.is_available()` is
  `False` here, so there is no way to run this on real GPU hardware, let
  alone claim a GPU throughput number. The entire reason speculative
  decoding helps in production (memory-bandwidth-bound decode, not
  compute-bound) can't be demonstrated or benchmarked honestly on CPU --
  see "Benchmark results" above for what the CPU numbers actually show
  instead.
- **The paged attention op is gather-then-matmul, not a fused kernel.**
  Real PagedAttention (vLLM) is a custom CUDA kernel that reads
  non-contiguous KV blocks directly during the attention computation.
  `paged_cache.py` instead gathers each request's blocks into a
  contiguous tensor before calling ordinary batched matmul attention --
  functionally equivalent and necessary for correctness testing on CPU,
  but it doesn't capture the memory-bandwidth savings a real fused
  kernel gets from never materializing the gathered tensor. This
  tradeoff is documented in the module's own docstring, mirrored here
  because it matters for the benchmark numbers above.
- **TinyGPT is intentionally tiny** (same model used in
  `inference-batching-lab`/`speculative-decoding-lab`): a few layers,
  small `d_model`, random-initialized, no real pretrained weights (no
  network access to download one in this sandbox). It's sufficient to
  prove the engine's correctness and to measure real call-overhead
  tradeoffs, but it is not a claim about any real model's behavior.
- **The "floor" draft model is maximally pessimistic.** An independent,
  randomly initialized draft model agrees with the target at roughly the
  1/vocab_size floor -- real production draft models (distilled or
  quantized versions of the target) are deliberately trained to agree
  far more often than that. The floor number here is a true worst case,
  not a representative one.

## What I'd build next

- A real fused paged-attention kernel (Triton, not CUDA by hand) to
  actually capture the memory-bandwidth savings paging is supposed to
  deliver, and to make the benchmark numbers in this README meaningful
  on real hardware instead of CPU call-overhead-bound.
- A trained (even lightly distilled) draft model instead of an
  independent random one, to get a realistic middle data point between
  this repo's floor and ceiling.
- Tree-based speculative decoding (multiple draft continuations per
  step, not a single linear chain of k tokens) -- a natural extension
  once per-request variable-length paging already works, and the next
  real complexity jump in the production systems this repo is modeling.
- Prefix caching across requests that share a prompt prefix (common in
  real serving workloads: system prompts, few-shot templates) -- paging
  already has the block-table structure to make shared, copy-on-write
  blocks possible without a redesign.
