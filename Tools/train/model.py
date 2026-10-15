"""Loom's transformer in MLX, for training on the Mac.

A small modern decoder (the Llama family's shape), written out by hand so every operation has
an exact twin in the Swift engine (LoomEngine/Model.swift) and in the float64 NumPy reference (ref.py):

  x = wte[token]                                         no position vector: RoPE turns q and k instead
  for each block:
      h = rms(x)·g1;  q = h·Wq, k = h·Wk, v = h·Wv        4 heads of 32
      q, k = rope(q, pos), rope(k, pos)                  each pair (2i, 2i+1) turns by pos·θ_i, θ_i = 10000^(−2i/32)
      x += softmax(q·kᵀ/√32 + causal mask)·v · Wo
      h = rms(x)·g2;  x += (silu(h·Wg) ⊙ (h·Wu)) · Wd    SwiGLU, 384 hidden "neurons"
  logits = rms(x)·gf · wteᵀ                              tied: the readout compares against the token vectors

rms(x) = x / sqrt(mean(x²) + 1e-5). No biases anywhere.
LoRA: an adapted matrix W becomes W + (α/r)·A·B with A (in×r), B (r×out); B starts at zero.
"""
import math
from dataclasses import dataclass, asdict

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    vocab: int = 4096
    ctx: int = 128
    d: int = 128
    heads: int = 4
    layers: int = 4
    ff: int = 384
    rope_base: float = 10000.0

    def dict(self):
        return asdict(self)


MATRICES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


def rms_norm(x, g, eps=1e-5):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * g


def silu(x):
    return x * mx.sigmoid(x)


def rope_tables(T, dh, base):
    inv = base ** (-mx.arange(0, dh, 2, dtype=mx.float32) / dh)           # dh/2 frequencies
    ang = mx.arange(T, dtype=mx.float32)[:, None] * inv[None, :]          # T × dh/2
    return mx.cos(ang), mx.sin(ang)


def rope(x, cos, sin):
    """x: B, H, T, dh. Rotates each adjacent pair (x[2i], x[2i+1]) by angle pos·θ_i."""
    B, H, T, dh = x.shape
    p = x.reshape(B, H, T, dh // 2, 2)
    a, b = p[..., 0], p[..., 1]
    ra = a * cos - b * sin
    rb = a * sin + b * cos
    return mx.stack([ra, rb], axis=-1).reshape(B, H, T, dh)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        s = 0.02
        r = 0.02 / math.sqrt(2 * c.layers)
        self.g1 = mx.ones((c.d,))
        self.g2 = mx.ones((c.d,))
        self.wq = mx.random.normal((c.d, c.d)) * s
        self.wk = mx.random.normal((c.d, c.d)) * s
        self.wv = mx.random.normal((c.d, c.d)) * s
        self.wo = mx.random.normal((c.d, c.d)) * r
        self.wg = mx.random.normal((c.d, c.ff)) * s
        self.wu = mx.random.normal((c.d, c.ff)) * s
        self.wd = mx.random.normal((c.ff, c.d)) * r

    def mm(self, x, name, lora):
        y = x @ getattr(self, name)
        if lora is not None and name in lora:
            a, b, scale = lora[name]
            y = y + scale * ((x @ a) @ b)
        return y

    def __call__(self, x, mask, cos, sin, lora=None):
        c = self.c
        B, T, _ = x.shape
        dh = c.d // c.heads
        h = rms_norm(x, self.g1)
        q = self.mm(h, "wq", lora).reshape(B, T, c.heads, dh).transpose(0, 2, 1, 3)
        k = self.mm(h, "wk", lora).reshape(B, T, c.heads, dh).transpose(0, 2, 1, 3)
        v = self.mm(h, "wv", lora).reshape(B, T, c.heads, dh).transpose(0, 2, 1, 3)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        att = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(dh)) + mask, axis=-1)
        a = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, c.d)
        x = x + self.mm(a, "wo", lora)
        h = rms_norm(x, self.g2)
        x = x + self.mm(silu(self.mm(h, "wg", lora)) * self.mm(h, "wu", lora), "wd", lora)
        return x


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.wte = mx.random.normal((c.vocab, c.d)) * 0.02
        self.blocks = [Block(c) for _ in range(c.layers)]
        self.gf = mx.ones((c.d,))

    def __call__(self, idx, lora=None):
        B, T = idx.shape
        x = self.wte[idx]
        cos, sin = rope_tables(T, self.c.d // self.c.heads, self.c.rope_base)
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, cos, sin, lora[i] if lora else None)
        return rms_norm(x, self.gf) @ self.wte.T


class LoRA(nn.Module):
    """Low-rank adapter on every matrix of every block."""

    def __init__(self, c: Config, rank=8, alpha=16.0, targets=MATRICES):
        super().__init__()
        self.rank, self.alpha, self.targets = rank, alpha, tuple(targets)
        dims = {"wq": (c.d, c.d), "wk": (c.d, c.d), "wv": (c.d, c.d), "wo": (c.d, c.d),
                "wg": (c.d, c.ff), "wu": (c.d, c.ff), "wd": (c.ff, c.d)}
        self.layers = []
        for _ in range(c.layers):
            layer = {}
            for name in self.targets:
                n_in, n_out = dims[name]
                layer[name + "_a"] = mx.random.normal((n_in, rank)) * (1.0 / math.sqrt(n_in))
                layer[name + "_b"] = mx.zeros((rank, n_out))
            self.layers.append(layer)

    def triples(self, strength=1.0):
        s = strength * self.alpha / self.rank
        return [{n: (l[n + "_a"], l[n + "_b"], s) for n in self.targets} for l in self.layers]
