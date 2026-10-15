"""Float64 NumPy reference for everything the Swift engine computes.

Independent of MLX: it reads an exported .loom file and recomputes the forward pass with every
intermediate kept, plus the knob transforms (sampling chain, quantisation, LoRA mixing, steering,
head muting, floor bypass, neuron clamps, logit lens). fixtures.py turns its outputs into JSON the
Swift parity tests check. The architecture is documented in model.py.
"""
import math

import numpy as np

from loomio import read_loom

MATRICES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


class Weights:
    def __init__(self, path):
        self.meta, t = read_loom(path)
        self.t = {k: v.astype(np.float64) for k, v in t.items()}
        c = self.meta["config"]
        self.vocab, self.ctx, self.d, self.heads, self.layers, self.ff = (
            c["vocab"], c["ctx"], c["d"], c["heads"], c["layers"], c["ff"])
        self.rope_base = c.get("rope_base", 10000.0)

    def __getitem__(self, k):
        return self.t[k]


def rms_norm(x, g, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_angles(T, dh, base, start=0):
    inv = base ** (-np.arange(0, dh, 2, dtype=np.float64) / dh)
    return (np.arange(start, start + T, dtype=np.float64)[:, None]) * inv[None, :]     # T × dh/2


def rope(x, ang):
    """x: H, T, dh."""
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def quantize(w, bits):
    """Symmetric per-output-column round-to-nearest (columns of an in×out matrix).
    bits >= 16 returns w unchanged; bits == 1 keeps sign × mean |w| of the column."""
    if bits >= 16:
        return w
    if bits == 1:
        return np.sign(w) * np.abs(w).mean(axis=0, keepdims=True)
    levels = 2 ** (bits - 1) - 1
    scale = np.abs(w).max(axis=0, keepdims=True) / levels
    scale[scale == 0] = 1
    return np.clip(np.round(w / scale), -levels, levels) * scale


def forward(W, ids, knobs=None, adapters=None):
    """knobs:
         bits     weight precision (16 = untouched)
         muted    set of (layer, head) whose output is zeroed before Wo
         bypass   set of layers skipped entirely (the residual passes straight through)
         steer    list of (layer, vector, strength) added to the residual after that layer
         clamp    list of (layer, neuron, value) fixing a SwiGLU neuron's activation
       adapters: list of (tensors, strength, alpha, rank) added as Σ strength·(α/r)·A·B"""
    k = knobs or {}
    bits = k.get("bits", 16)
    muted, bypass = k.get("muted", set()), k.get("bypass", set())
    steer, clamp = k.get("steer", []), k.get("clamp", [])
    T, dh = len(ids), W.d // W.heads
    wte = quantize(W["wte"].T, bits).T
    x = wte[ids].copy()
    out = {"x0": x.copy(), "layers": []}
    mask = np.triu(np.full((T, T), -np.inf), 1)
    ang = rope_angles(T, dh, W.rope_base)

    def mm(h, l, name):
        y = h @ quantize(W[f"b{l}.{name}"], bits)
        for lora, strength, alpha, rank in adapters or []:
            key = f"b{l}.{name}"
            if key + "_a" in lora:
                y = y + strength * (alpha / rank) * ((h @ lora[key + "_a"]) @ lora[key + "_b"])
        return y

    for l in range(W.layers):
        L = {"in": x.copy()}
        if l in bypass:
            L.update(skipped=True, out=x.copy())
            out["layers"].append(L)
            continue
        h = rms_norm(x, W[f"b{l}.g1"])
        q = mm(h, l, "wq").reshape(T, W.heads, dh).transpose(1, 0, 2)
        kk = mm(h, l, "wk").reshape(T, W.heads, dh).transpose(1, 0, 2)
        v = mm(h, l, "wv").reshape(T, W.heads, dh).transpose(1, 0, 2)
        q, kk = rope(q, ang), rope(kk, ang)
        s = q @ kk.transpose(0, 2, 1) / math.sqrt(dh) + mask
        s = s - s.max(-1, keepdims=True)
        att = np.exp(s)
        att /= att.sum(-1, keepdims=True)
        heads = att @ v                                       # H, T, dh
        for (ml, mh) in muted:
            if ml == l:
                heads[mh] = 0
        a = heads.transpose(1, 0, 2).reshape(T, W.d)
        attn_out = mm(a, l, "wo")
        x = x + attn_out
        L.update(h1=h, q=q, k=kk, v=v, att=att, heads=heads, attn_out=attn_out, mid=x.copy())
        h2 = rms_norm(x, W[f"b{l}.g2"])
        gate, up = mm(h2, l, "wg"), mm(h2, l, "wu")
        act = silu(gate) * up
        for (cl, n, value) in clamp:
            if cl == l:
                act[:, n] = value
        mlp_out = mm(act, l, "wd")
        x = x + mlp_out
        for (sl, vec, strength) in steer:
            if sl == l:
                x = x + strength * vec
        L.update(h2=h2, gate=gate, up=up, act=act, mlp_out=mlp_out, out=x.copy())
        out["layers"].append(L)
    out["final"] = rms_norm(x, W["gf"])
    out["logits"] = out["final"] @ wte.T
    return out


def lens(W, resid):
    """Logit lens: read a residual vector through the final norm and the tied readout."""
    return rms_norm(resid, W["gf"]) @ W["wte"].T


def softmax(z):
    z = z - z[np.isfinite(z)].max()
    e = np.where(np.isfinite(z), np.exp(z), 0.0)
    return e / e.sum()


def sampling_distribution(logits, context, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repetition=1.0, bias=None):
    """The app's sampler chain, in this order:
       logit bias -> repetition penalty -> temperature -> softmax -> top-k -> top-p -> min-p -> renormalise.
    temperature 0 is greedy. Returns (final probabilities, probabilities after temperature only)."""
    z = logits.astype(np.float64).copy()
    for t, b in (bias or {}).items():
        z[t] += b
    if repetition != 1.0:
        for t in set(context):
            z[t] = z[t] / repetition if z[t] > 0 else z[t] * repetition
    if temperature <= 0:
        p = np.zeros_like(z)
        p[int(np.argmax(z))] = 1.0
        return p, p.copy()
    p = softmax(z / temperature)
    heat = p.copy()
    keep = np.ones_like(p, dtype=bool)
    order = np.argsort(-p, kind="stable")
    if top_k > 0:
        keep[order[top_k:]] = False
    if top_p < 1.0:
        cum = np.cumsum(p[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        keep[order[cut:]] = False
    if min_p > 0:
        keep &= p >= min_p * p.max()
    keep[order[0]] = True
    q = np.where(keep, p, 0.0)
    return q / q.sum(), heat
