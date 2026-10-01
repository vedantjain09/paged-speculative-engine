"""The one primitive every phase of this engine's forward computation goes
through: gather a batch of requests' paged KV state, run one ordinary
(non-paged) model forward call appending exactly W new tokens for every
row -- W must be uniform across the batch, since a single tensor can't
vary its width per row -- and write the W newly-computed positions back
into each request's own slot in the paged cache.

Uniform W is not a limitation specific to this repo: it's true of every
real batched inference engine too. The reason speculative decoding looks
like it breaks batching is that people conflate "forward-call width" with
"how many tokens end up kept." They're different things here: the forward
call is always uniform (drafting appends 1 token/row/round, verification
appends k tokens/row in one call), and the *variable* part -- different
requests accepting different numbers of those k drafted tokens -- is
handled entirely by each request's own cache.truncate() call afterward,
independent of every other request's. See speculative.py.
"""
from __future__ import annotations

import torch

from engine.model import TinyGPT
from engine.paged_cache import PagedKVCache


@torch.no_grad()
def batched_forward_append(
    model: TinyGPT,
    cache: PagedKVCache,
    request_ids: list[str],
    new_input_ids: torch.Tensor,
) -> torch.Tensor:
    """new_input_ids: [batch, W], W new tokens for every request in
    request_ids (same W for all rows). Returns logits [batch, W, vocab].
    Also appends the W newly-computed KV positions into each request's
    own slot in `cache`."""
    k_out, v_out, mask, lengths = cache.gather(request_ids)
    past_kv = [(k_out[layer], v_out[layer]) for layer in range(cache.n_layers)]
    max_cached_len = mask.shape[1]
    batch = len(request_ids)
    width = new_input_ids.shape[1]

    # The W new positions are always real content, never padding.
    decode_pad_mask = torch.cat([mask, torch.ones(batch, width, dtype=torch.bool)], dim=1)

    # Same fix as inference-batching-lab's scheduler (bug #6 in that
    # README): the model always writes new tokens at raw tensor index
    # max_cached_len.. for *every* row (torch.cat always appends at the
    # same column), but each row's own *true* position for those columns
    # is that row's own cached length + offset, which is smaller than
    # max_cached_len for any row shorter than the batch's longest. Every
    # other (padded) column keeps its raw index as its position (correct,
    # since real content is left-aligned / right-padded); only the W new
    # columns need the override.
    key_positions = torch.arange(max_cached_len + width).unsqueeze(0).expand(batch, -1).clone()
    for w in range(width):
        key_positions[:, max_cached_len + w] = lengths + w

    logits, new_kv = model(
        new_input_ids,
        past_kv=past_kv,
        position_offset=lengths,
        padding_mask=decode_pad_mask,
        key_positions=key_positions,
    )

    for i, rid in enumerate(request_ids):
        new_k_per_layer = [new_kv[layer][0][i, :, max_cached_len : max_cached_len + width, :] for layer in range(cache.n_layers)]
        new_v_per_layer = [new_kv[layer][1][i, :, max_cached_len : max_cached_len + width, :] for layer in range(cache.n_layers)]
        cache.append(rid, new_k_per_layer, new_v_per_layer)

    return logits
