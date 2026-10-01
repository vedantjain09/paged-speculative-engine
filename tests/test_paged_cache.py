"""Unit tests for BlockAllocator and PagedKVCache."""
from __future__ import annotations

import torch

from engine.paged_cache import BlockAllocator, PagedKVCache


def test_allocator_allocate_free_roundtrip():
    alloc = BlockAllocator(num_blocks=4)
    assert alloc.num_free == 4
    a = alloc.allocate()
    b = alloc.allocate()
    assert alloc.num_free == 2
    assert a != b
    alloc.free(a)
    assert alloc.num_free == 3
    alloc.free(b)
    assert alloc.num_free == 4


def test_allocator_raises_when_exhausted():
    alloc = BlockAllocator(num_blocks=2)
    alloc.allocate()
    alloc.allocate()
    try:
        alloc.allocate()
        assert False, "expected MemoryError"
    except MemoryError:
        pass


def test_allocator_raises_on_double_free():
    alloc = BlockAllocator(num_blocks=2)
    block = alloc.allocate()
    alloc.free(block)
    try:
        alloc.free(block)
        assert False, "expected ValueError on double free"
    except ValueError:
        pass


def _mk_kv(n_layers: int, n_heads: int, num_new: int, head_dim: int, fill: float) -> list[torch.Tensor]:
    return [torch.full((n_heads, num_new, head_dim), fill) for _ in range(n_layers)]


def test_append_single_block_fits_exactly():
    cache = PagedKVCache(n_layers=2, n_heads=2, head_dim=4, block_size=4, num_blocks=8)
    cache.create_request("r1")
    k = _mk_kv(2, 2, num_new=3, head_dim=4, fill=1.0)
    cache.append("r1", k, k)
    assert cache.length("r1") == 3
    assert cache.num_blocks_in_use("r1") == 1

    k_out, v_out, mask, lengths = cache.gather(["r1"])
    assert lengths.tolist() == [3]
    assert mask.tolist() == [[True, True, True]]
    assert torch.allclose(k_out[0][0], torch.full((2, 3, 4), 1.0))


def test_append_spans_multiple_blocks():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=8)
    cache.create_request("r1")
    k = _mk_kv(1, 1, num_new=10, head_dim=2, fill=2.0)
    cache.append("r1", k, k)
    assert cache.length("r1") == 10
    # ceil(10/4) = 3 blocks
    assert cache.num_blocks_in_use("r1") == 3

    k_out, v_out, mask, lengths = cache.gather(["r1"])
    assert lengths.tolist() == [10]
    assert mask.sum().item() == 10
    assert torch.allclose(k_out[0][0, :, :10, :], torch.full((1, 10, 2), 2.0))


def test_append_across_multiple_calls_crossing_block_boundary():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=8)
    cache.create_request("r1")
    cache.append("r1", _mk_kv(1, 1, 3, 2, 5.0), _mk_kv(1, 1, 3, 2, 5.0))  # fills 3/4 of block 0
    cache.append("r1", _mk_kv(1, 1, 3, 2, 6.0), _mk_kv(1, 1, 3, 2, 6.0))  # crosses into block 1

    assert cache.length("r1") == 6
    k_out, _, mask, lengths = cache.gather(["r1"])
    assert lengths.item() == 6
    expected = torch.tensor([5.0, 5.0, 5.0, 6.0, 6.0, 6.0]).view(1, 6, 1).expand(1, 6, 2)
    assert torch.allclose(k_out[0][0, :, :6, :], expected)


def test_truncate_frees_trailing_blocks():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=4)
    cache.create_request("r1")
    cache.append("r1", _mk_kv(1, 1, 8, 2, 1.0), _mk_kv(1, 1, 8, 2, 1.0))  # 2 full blocks
    assert cache.num_blocks_in_use("r1") == 2
    assert cache.allocator.num_free == 2

    cache.truncate("r1", 3)  # back into the first block
    assert cache.length("r1") == 3
    assert cache.num_blocks_in_use("r1") == 1
    assert cache.allocator.num_free == 3  # the second block was returned to the pool


def test_truncate_exact_block_boundary_keeps_full_blocks():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=4)
    cache.create_request("r1")
    cache.append("r1", _mk_kv(1, 1, 8, 2, 1.0), _mk_kv(1, 1, 8, 2, 1.0))  # 2 full blocks

    cache.truncate("r1", 4)  # exactly one full block's worth
    assert cache.num_blocks_in_use("r1") == 1
    assert cache.allocator.num_free == 3


def test_truncate_to_zero_frees_all_blocks():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=4)
    cache.create_request("r1")
    cache.append("r1", _mk_kv(1, 1, 6, 2, 1.0), _mk_kv(1, 1, 6, 2, 1.0))
    cache.truncate("r1", 0)
    assert cache.num_blocks_in_use("r1") == 0
    assert cache.allocator.num_free == 4


def test_free_request_returns_all_blocks():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=4)
    cache.create_request("r1")
    cache.append("r1", _mk_kv(1, 1, 7, 2, 1.0), _mk_kv(1, 1, 7, 2, 1.0))
    assert cache.allocator.num_free == 2
    cache.free_request("r1")
    assert cache.allocator.num_free == 4


def test_two_requests_do_not_alias_physical_blocks():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=8)
    cache.create_request("a")
    cache.create_request("b")

    # interleave writes so any accidental shared-state bug would show up
    cache.append("a", _mk_kv(1, 1, 2, 2, 10.0), _mk_kv(1, 1, 2, 2, 10.0))
    cache.append("b", _mk_kv(1, 1, 2, 2, 20.0), _mk_kv(1, 1, 2, 2, 20.0))
    cache.append("a", _mk_kv(1, 1, 5, 2, 10.0), _mk_kv(1, 1, 5, 2, 10.0))
    cache.append("b", _mk_kv(1, 1, 5, 2, 20.0), _mk_kv(1, 1, 5, 2, 20.0))

    k_out, _, mask, lengths = cache.gather(["a", "b"])
    assert lengths.tolist() == [7, 7]
    assert torch.allclose(k_out[0][0, :, :7, :], torch.full((1, 7, 2), 10.0))
    assert torch.allclose(k_out[0][1, :, :7, :], torch.full((1, 7, 2), 20.0))


def test_gather_pads_requests_of_different_lengths():
    cache = PagedKVCache(n_layers=1, n_heads=1, head_dim=2, block_size=4, num_blocks=8)
    cache.create_request("short")
    cache.create_request("long")
    cache.append("short", _mk_kv(1, 1, 2, 2, 1.0), _mk_kv(1, 1, 2, 2, 1.0))
    cache.append("long", _mk_kv(1, 1, 6, 2, 2.0), _mk_kv(1, 1, 6, 2, 2.0))

    _, _, mask, lengths = cache.gather(["short", "long"])
    assert lengths.tolist() == [2, 6]
    assert mask[0].tolist() == [True, True, False, False, False, False]
    assert mask[1].tolist() == [True] * 6
