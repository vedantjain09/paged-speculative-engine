"""Property-based tests for PagedKVCache: replay random sequences of
create/append/truncate/free operations across several concurrent requests
and check the invariants that actually matter for correctness -- block
conservation (no block is ever owned by two requests, none are lost) and
exact content fidelity (what gather() returns matches what was written,
not some other request's data)."""
from __future__ import annotations

import torch
from hypothesis import assume, given, settings, strategies as st

from engine.paged_cache import PagedKVCache

BLOCK_SIZE = 4
NUM_BLOCKS = 16
N_LAYERS = 1
N_HEADS = 1
HEAD_DIM = 2
REQUEST_IDS = ["r0", "r1", "r2"]


def _kv(num_new: int, fill: float) -> list[torch.Tensor]:
    return [torch.full((N_HEADS, num_new, HEAD_DIM), fill) for _ in range(N_LAYERS)]


# Ops: ("append", rid, num_new) | ("truncate", rid, new_length) | ("free", rid)
op_strategy = st.one_of(
    st.tuples(st.just("append"), st.sampled_from(REQUEST_IDS), st.integers(min_value=1, max_value=5)),
    st.tuples(st.just("truncate"), st.sampled_from(REQUEST_IDS), st.integers(min_value=0, max_value=20)),
    st.tuples(st.just("free"), st.sampled_from(REQUEST_IDS), st.just(0)),
)


@settings(max_examples=150, deadline=None)
@given(ops=st.lists(op_strategy, min_size=1, max_size=40))
def test_random_operation_sequences_preserve_invariants(ops):
    cache = PagedKVCache(N_LAYERS, N_HEADS, HEAD_DIM, BLOCK_SIZE, NUM_BLOCKS)
    # Ground truth: for each request, the exact sequence of fill-values
    # written and still logically present (truncate drops the tail).
    ground_truth: dict[str, list[float]] = {}
    alive: set[str] = set()
    fill_counter = [100.0]

    def active_block_total() -> int:
        return sum(cache.num_blocks_in_use(rid) for rid in alive)

    for kind, rid, arg in ops:
        if kind == "append":
            if rid not in alive:
                cache.create_request(rid)
                ground_truth[rid] = []
                alive.add(rid)
            fill_counter[0] += 1.0
            fill = fill_counter[0]
            try:
                cache.append(rid, _kv(arg, fill), _kv(arg, fill))
            except MemoryError:
                # Pool exhaustion is a legitimate outcome of a fixed-size
                # pool under a large enough random append sequence -- not
                # a logic bug, and not what this test's invariants are
                # about (see test_allocator_raises_when_exhausted for the
                # dedicated exhaustion-behavior test). Discard this
                # example rather than let it look like a failure.
                assume(False)
            ground_truth[rid].extend([fill] * arg)

        elif kind == "truncate":
            if rid in alive:
                new_length = min(arg, cache.length(rid))
                cache.truncate(rid, new_length)
                ground_truth[rid] = ground_truth[rid][:new_length]

        elif kind == "free":
            if rid in alive:
                cache.free_request(rid)
                alive.discard(rid)
                del ground_truth[rid]

        # Invariant 1: block conservation -- every block is either free or
        # owned by exactly one alive request; none are lost or double-counted.
        assert cache.allocator.num_free + active_block_total() == NUM_BLOCKS

    # Invariant 2: content fidelity -- gather() returns exactly what each
    # alive request's own ground truth says, never another request's data
    # (which is what a block-aliasing bug would produce) and never stale
    # data from before a truncate (which is what a free-but-don't-clear
    # bug would produce).
    if alive:
        ordered = sorted(alive)
        k_out, v_out, mask, lengths = cache.gather(ordered)
        for i, rid in enumerate(ordered):
            expected_len = len(ground_truth[rid])
            assert lengths[i].item() == expected_len
            if expected_len > 0:
                expected = torch.tensor(ground_truth[rid]).view(1, expected_len, 1).expand(N_HEADS, expected_len, HEAD_DIM)
                assert torch.allclose(k_out[0][i, :, :expected_len, :], expected)
                assert torch.allclose(v_out[0][i, :, :expected_len, :], expected)
            assert mask[i, :expected_len].all()
            if mask.shape[1] > expected_len:
                assert not mask[i, expected_len:].any()
