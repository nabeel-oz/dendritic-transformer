"""Shared decoder-only transformer backbone.

Plain-language purpose: a small GPT. Embed characters, run them through N
identical blocks of (causal self-attention + FFN), and predict the next
character. Everything here is held FIXED across the experiment's variants; the
ONLY moving part is which FFN `build_ffn(cfg)` returns. The backbone never
branches on the variant — that is what makes the comparison fair.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffn import build_ffn


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention (each position attends only leftward)."""

    def __init__(self, d_model: int, n_head: int, context: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash/SDPA causal attention (efficient, fits 8 GB comfortably).
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)); x + ffn(norm(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_head, cfg.context, cfg.dropout
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = build_ffn(cfg)  # the only swappable piece

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.context = cfg.context

        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # weight tying: input embedding and output projection share weights
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, trainable_only: bool = True) -> int:
        ps = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        # subtract tied head (shares tok_emb storage) to avoid double counting
        n = sum(p.numel() for p in ps)
        return n - self.head.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.context, f"sequence length {T} > context {self.context}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None):
        """Autoregressively sample characters from a context prompt."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
