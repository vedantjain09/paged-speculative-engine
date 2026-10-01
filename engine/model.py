"""A minimal decoder-only transformer, implemented from scratch (no pretrained
weights -- see README "Honest limitations" for why), with explicit per-layer KV-cache
support written out by hand rather than relying on a library's built-in generation
loop. The point of this repo is the serving/batching layer underneath it, and that
layer needs real control over exactly what gets recomputed at each decode step, which
`nn.TransformerDecoder` + `model.generate()` hide behind an opaque loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# KV cache for one layer: (key, value), each shaped [batch, heads, seq_len, head_dim]
LayerKV = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 512


def build_causal_mask(
    seq_len: int,
    total_len: int,
    position_offset: int | torch.Tensor,
    device: torch.device,
    key_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """mask[i, j] (or mask[b, i, j] when position_offset is per-batch-item) is True
    iff the query at absolute position (position_offset + i) is allowed to attend to
    the key at absolute position j (0 <= j < total_len).

    This has to be computed from absolute positions, not from seq_len/total_len shape
    alone: during an incremental decode step seq_len is 1 (a single new query) while
    total_len is the full cached length so far, so a naive torch.tril(ones(seq_len,
    total_len)) only allows attending to key index 0 instead of the whole cache --
    see the "real bug" writeup in the README for exactly this failure.

    position_offset is a per-batch-item tensor ([batch]) in the continuous-batching
    scheduler, where different requests in the same batched forward call can have
    different true sequence lengths (each was prefilled independently before joining
    the batch) -- a single scalar offset can't express that.
    """
    if key_positions is None:
        key_positions = torch.arange(total_len, device=device)  # [total_len]

    if isinstance(position_offset, torch.Tensor):
        step = torch.arange(seq_len, device=device)  # [seq_len]
        query_positions = position_offset.view(-1, 1) + step.view(1, -1)  # [batch, seq_len]
        if key_positions.dim() == 1:
            key_positions = key_positions.view(1, -1).expand(position_offset.shape[0], -1)  # [batch, total_len]
        return query_positions.unsqueeze(-1) >= key_positions.unsqueeze(1)  # [batch, seq_len, total_len]

    query_positions = torch.arange(position_offset, position_offset + seq_len, device=device)
    return query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)  # [seq_len, total_len]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: LayerKV | None,
        position_offset: int | torch.Tensor,
        padding_mask: torch.Tensor | None,
        key_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv: LayerKV = (k, v)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        total_len = k.shape[2]

        causal = build_causal_mask(seq_len, total_len, position_offset, x.device, key_positions)
        if causal.dim() == 3:  # per-batch-item offsets: [batch, seq_len, total_len]
            mask = causal.view(batch, 1, seq_len, total_len)
        else:  # single shared offset: [seq_len, total_len], broadcasts over batch
            mask = causal.view(1, 1, seq_len, total_len)
        if padding_mask is not None:
            # padding_mask: [batch, total_len], True where the key position is a real
            # (non-padding) token. Combined per-batch-item so a padded slot in one
            # request in the batch can never be attended to, even by real tokens in
            # other requests sharing the same batched forward pass.
            mask = mask & padding_mask.view(batch, 1, 1, total_len)

        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).reshape(batch, seq_len, self.n_heads * self.head_dim)
        return self.out(attended), new_kv


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: LayerKV | None,
        position_offset: int | torch.Tensor,
        padding_mask: torch.Tensor | None,
        key_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        attn_out, new_kv = self.attn(self.ln1(x), past_kv, position_offset, padding_mask, key_positions)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv


class TinyGPT(nn.Module):
    """A from-scratch GPT-style decoder. Untrained/random weights -- see README; this
    repo benchmarks serving-layer throughput and latency, not generation quality."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: list[LayerKV] | None = None,
        position_offset: int | torch.Tensor = 0,
        padding_mask: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[LayerKV]]:
        """input_ids: [batch, seq_len]. If past_kv is given, seq_len is normally 1 (a
        single new decode step) and position_offset must equal the cached length so
        positional embeddings and the causal mask continue from the right position.

        padding_mask (optional): [batch, total_len_after_this_call] bool, True for
        real tokens. Only needed for batched requests of different lengths (see
        scheduler.py); a single unpadded sequence never needs it.
        """
        batch, seq_len = input_ids.shape
        if isinstance(position_offset, torch.Tensor):
            step = torch.arange(seq_len, device=input_ids.device)
            positions = position_offset.view(-1, 1) + step.view(1, -1)  # [batch, seq_len]
            x = self.token_emb(input_ids) + self.pos_emb(positions)
        else:
            positions = torch.arange(position_offset, position_offset + seq_len, device=input_ids.device)
            x = self.token_emb(input_ids) + self.pos_emb(positions).unsqueeze(0)

        # Real bug found via test_static_batching_matches_sequential_generation: a
        # left-padded position's own query can end up with *every* key masked out
        # (its only causally-visible key is itself, which is also the padding
        # position being masked as invalid) -- softmax over an all -inf row is NaN,
        # and NaN * 0.0 (the padding mask's attention weight for that key) is still
        # NaN, not 0, so it silently poisons every later position's key/value once
        # this padding position is cached. Zeroing the padding positions' hidden
        # state after every block keeps that NaN from ever being produced, instead of
        # trying to mask it out after the fact.
        valid = padding_mask[:, -seq_len:].unsqueeze(-1) if padding_mask is not None else None
        if valid is not None:
            x = x * valid

        new_past_kv: list[LayerKV] = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, layer_new_kv = block(x, layer_past, position_offset, padding_mask, key_positions)
            if valid is not None:
                x = x * valid
            new_past_kv.append(layer_new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_past_kv
