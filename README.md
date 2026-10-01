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

## Status

Actively built section by section -- see commit history. This section and
the numbers below are filled in as each part lands; do not trust numbers
that look like placeholders.

## What I'd build next

(filled in at the end, honestly, same as my other serving project)
