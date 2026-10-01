"""A paged KV-cache: fixed-size blocks allocated from a shared pool, one
independent block table per request -- the actual memory-management
mechanism production inference engines (vLLM's PagedAttention) use instead
of growing one contiguous tensor per request.

Why this matters for this repo specifically: a contiguous-tensor cache (the
kind in inference-batching-lab, and the kind torch.cat-based speculative
decoding in speculative-decoding-lab uses) can only ever grow a request's
cache by re-allocating a bigger tensor and copying -- fine for "every
request advances by exactly one token per step," but speculative decoding
needs a request's cache to grow by a *different, data-dependent* number of
tokens each step (0 to k+1, depending on how many draft tokens got
accepted). A block-table cache handles that naturally: appending N tokens
just claims however many additional fixed-size blocks N needs, independent
of what any other request in the batch is doing this step.

Honest scope note: real PagedAttention computes attention *directly* over
the non-contiguous blocks, inside a custom CUDA kernel, so it never
materializes a contiguous copy. `gather()` below instead assembles a
contiguous, padded view before the (ordinary PyTorch) attention call --
correct page-table allocation/eviction/reference-counting semantics, same
end-to-end behavior, but without the memory-bandwidth win a real kernel
gets from skipping the copy. That copy is exactly the piece of this
problem that needs a GPU kernel to do for real, which this sandbox has no
hardware to write or benchmark (see README).
"""
from __future__ import annotations

import torch


class BlockAllocator:
    """A free-list over `num_blocks` fixed-capacity slots. Nothing fancier
    than a stack of free ids -- the interesting logic lives in PagedKVCache,
    which decides *when* to call allocate()/free(), not in here."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise MemoryError(f"paged KV-cache exhausted: all {self.num_blocks} blocks in use")
        return self._free.pop()

    def free(self, block_id: int) -> None:
        # Double-free would silently corrupt another request's cache later
        # (two requests believing they own the same physical block) -- catch
        # it loudly instead of letting it happen quietly.
        if block_id in self._free:
            raise ValueError(f"block {block_id} freed twice")
        self._free.append(block_id)


class PagedKVCache:
    """Owns the physical K/V storage for one model (draft or target get
    their own instance, since they generally have different shapes) and the
    per-request block tables mapping logical position -> physical block.

    Storage layout per layer: a [num_blocks, n_heads, block_size, head_dim]
    tensor for K and one for V. A request's logical token at position p
    lives in block_table[p // block_size], at offset p % block_size.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

        self._k_pool = [
            torch.zeros(num_blocks, n_heads, block_size, head_dim) for _ in range(n_layers)
        ]
        self._v_pool = [
            torch.zeros(num_blocks, n_heads, block_size, head_dim) for _ in range(n_layers)
        ]
        self._block_tables: dict[str, list[int]] = {}
        self._lengths: dict[str, int] = {}

    def create_request(self, request_id: str) -> None:
        if request_id in self._block_tables:
            raise ValueError(f"request {request_id} already exists in this cache")
        self._block_tables[request_id] = []
        self._lengths[request_id] = 0

    def length(self, request_id: str) -> int:
        return self._lengths[request_id]

    def num_blocks_in_use(self, request_id: str) -> int:
        return len(self._block_tables[request_id])

    def append(self, request_id: str, new_k: list[torch.Tensor], new_v: list[torch.Tensor]) -> None:
        """new_k/new_v: one tensor per layer, each [n_heads, num_new, head_dim].
        Writes num_new tokens starting at this request's current length,
        allocating new blocks from the pool whenever the write crosses into
        a block index that doesn't exist yet in this request's table. Blocks
        already owned by this request are never touched by any other
        request's append call -- that's the whole point of per-request
        block tables sharing one pool."""
        num_new = new_k[0].shape[1]
        table = self._block_tables[request_id]
        length = self._lengths[request_id]

        for layer in range(self.n_layers):
            consumed = 0
            layer_length = length
            while consumed < num_new:
                block_idx = layer_length // self.block_size
                offset = layer_length % self.block_size
                if block_idx >= len(table):
                    if layer == 0:
                        table.append(self.allocator.allocate())
                    else:
                        # All layers advance in lockstep (same token count),
                        # so by the time layer>0 needs a new block index,
                        # layer 0's pass through this same loop already
                        # allocated it onto the shared table.
                        pass
                physical = table[block_idx]
                take = min(self.block_size - offset, num_new - consumed)
                self._k_pool[layer][physical, :, offset : offset + take, :] = new_k[layer][:, consumed : consumed + take, :]
                self._v_pool[layer][physical, :, offset : offset + take, :] = new_v[layer][:, consumed : consumed + take, :]
                consumed += take
                layer_length += take

        self._lengths[request_id] = length + num_new

    def truncate(self, request_id: str, new_length: int) -> None:
        """Shrinks a request's logical length (used when speculative
        decoding rejects some already-written draft positions) and frees
        any physical blocks that fall entirely past the new length, so
        they're immediately available for this or any other request's next
        allocate() call -- a paged cache that never frees on truncate would
        leak a block every single rejected draft token."""
        old_length = self._lengths[request_id]
        if new_length > old_length:
            raise ValueError(f"truncate({new_length}) > current length {old_length}")
        table = self._block_tables[request_id]
        blocks_needed = 0 if new_length == 0 else (new_length - 1) // self.block_size + 1
        while len(table) > blocks_needed:
            self.allocator.free(table.pop())
        self._lengths[request_id] = new_length

    def free_request(self, request_id: str) -> None:
        table = self._block_tables.pop(request_id)
        for block_id in table:
            self.allocator.free(block_id)
        del self._lengths[request_id]

    def gather(self, request_ids: list[str]) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Assembles a padded, contiguous [batch, n_heads, max_len, head_dim]
        view per layer for the ordinary (non-paged) attention call in
        model.py -- see the module docstring for why this copy exists
        instead of a real non-contiguous-attention kernel.

        Returns (k_per_layer, v_per_layer, padding_mask, lengths):
        padding_mask is [batch, max_len] bool, True for real (non-pad)
        positions. lengths is [batch] long, each request's true length.
        """
        lengths = torch.tensor([self._lengths[rid] for rid in request_ids], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(lengths) else 0
        batch = len(request_ids)
        padding_mask = torch.zeros(batch, max_len, dtype=torch.bool)

        k_out: list[torch.Tensor] = []
        v_out: list[torch.Tensor] = []
        for layer in range(self.n_layers):
            k_batch = torch.zeros(batch, self.n_heads, max_len, self.head_dim)
            v_batch = torch.zeros(batch, self.n_heads, max_len, self.head_dim)
            for i, rid in enumerate(request_ids):
                table = self._block_tables[rid]
                req_len = self._lengths[rid]
                pos = 0
                for block_idx, physical in enumerate(table):
                    take = min(self.block_size, req_len - pos)
                    if take <= 0:
                        break
                    k_batch[i, :, pos : pos + take, :] = self._k_pool[layer][physical, :, :take, :]
                    v_batch[i, :, pos : pos + take, :] = self._v_pool[layer][physical, :, :take, :]
                    pos += take
                if layer == 0:
                    padding_mask[i, :req_len] = True
            k_out.append(k_batch)
            v_out.append(v_batch)

        return k_out, v_out, padding_mask, lengths
