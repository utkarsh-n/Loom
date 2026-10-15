#!/usr/bin/env python3
"""Mac-side data so the phone can run Qwen3-4B-Instruct-2507 (4-bit MLX) as Loom's engine.

  ~/Developer/cipher-llm/venv/bin/python precompute.py [catalogue] [reference] [steering]

With no arguments all three stages run, in that order, each writing its file as soon as it is done.
Outputs go to ~/Developer/loom-data/qwen/extras/ (override with QWEN_EXTRAS):

  qwen-catalogue.json   the 8,192 commonest corpus tokens (+ chat specials): PCA star map of their
                        embedding rows, each token's script and corpus frequency.
  qwen-reference.json   a float32 reference trace of two prompts (logits, attention rows, residual norms,
                        an MLP activation, logit lens) for checking the phone's Swift forward pass.
  qwen-steering.loom    difference-of-means steering directions after every one of the 36 layers.

Everything runs in float32 activations: the model's bf16 scales/biases/norm weights are widened to f32
(exact) before anything is computed, so 4-bit weights dequantize to the same values the phone gets.
The steering pass checkpoints to extras/.steering-partial.npz and resumes from it if interrupted.
"""
import json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "train"))
from loomio import write_loom  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

DATA = os.path.expanduser("~/Developer/loom-data")
MODEL_DIRS = [os.path.join(DATA, "qwen", "Qwen3-4B-Instruct-2507-4bit"),
              os.path.expanduser("~/Developer/cipher-llm/models/Qwen3-4B-Instruct-2507-4bit")]
MODEL_BYTES = 2_263_022_417
OUT = os.environ.get("QWEN_EXTRAS", os.path.join(DATA, "qwen", "extras"))
LINES = os.path.join(DATA, "lines.tsv")
HF_NAME = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

LANGS = ["en", "es", "fr", "hi", "gu"]
FAMILIES = ["dread", "tender", "wonder", "rousing", "wry"]
SPECIALS = [151643, 151644, 151645]          # <|endoftext|> <|im_start|> <|im_end|>
CATALOGUE_SIZE = 8192
PER_GROUP, EVERYONE, MAX_TOKENS, SEED, BATCH = 200, 400, 48, 17, 32

D, LAYERS, HEADS, KV_HEADS, HEAD_DIM, EPS, THETA = 2560, 36, 32, 8, 128, 1e-6, 5_000_000.0
REF_PROMPTS = {"raw": "The night was",
               "chat": "<|im_start|>user\nThe night was<|im_end|>\n<|im_start|>assistant\n"}
REF_LAYERS, REF_HEADS = [0, 17, 35], [0, 7, 31]
PROBE_IDS = [0, 1, 100, 1000, 10000, 151643, 151645]

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------------------------------------- setup

def model_dir():
    for d in MODEL_DIRS:
        f = os.path.join(d, "model.safetensors")
        if os.path.exists(f):
            for _ in range(720):                       # wait up to an hour for a copy in progress
                if os.path.getsize(f) == MODEL_BYTES:
                    return d
                log(f"waiting for {f} ({os.path.getsize(f):,} / {MODEL_BYTES:,} bytes)")
                time.sleep(5)
    raise SystemExit("no complete Qwen3-4B-Instruct-2507-4bit model found")


def read_corpus():
    rows = []
    with open(LINES, encoding="utf-8") as f:
        for row in f:
            s, lang, work, moods, text = row.rstrip("\n").split("\t", 4)
            rows.append((s, lang, work, [m for m in moods.split("|") if m], text))
    return rows


def encode_all(hf, texts, chunk=20000):
    out = []
    for i in range(0, len(texts), chunk):
        out.extend(hf(texts[i:i + chunk], add_special_tokens=False)["input_ids"])
    return out


def script_of(text):
    """Same rule as Tools/train/extras.py: the first character that names a script decides."""
    if "�" in text:
        return "mark"                                  # a partial UTF-8 byte sequence
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: return "devanagari"
        if 0x0A80 <= o <= 0x0AFF: return "gujarati"
        if ch.isalpha(): return "latin"
        if ch.isdigit(): return "digit"
    return "mark"


def deq(mod):
    """A quantized Linear/Embedding's weight matrix as float32 numpy, [out, in]."""
    w = mx.dequantize(mod.weight, mod.scales, mod.biases, group_size=mod.group_size, bits=mod.bits,
                      mode=getattr(mod, "mode", "affine"))
    return np.array(w.astype(mx.float32))


def f32(a):
    return np.array(a.astype(mx.float32))


# ------------------------------------------------------------------------------------------ catalogue

def catalogue(model, hf, rows, enc):
    log("catalogue: counting token frequencies over", len(rows), "lines")
    counts = np.bincount(np.fromiter((t for ids in enc for t in ids), dtype=np.int64), minlength=151936)
    order = np.argsort(-counts, kind="stable")[:CATALOGUE_SIZE]
    ids = sorted(set(order.tolist()) | set(SPECIALS))
    E = f32(model.model.embed_tokens(mx.array(ids))).astype(np.float64)
    mean = E.mean(0)
    X = E - mean
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    for i in range(3):                                 # deterministic sign: heaviest loading positive
        if Vt[i][np.argmax(np.abs(Vt[i]))] < 0:
            Vt[i] = -Vt[i]
    P = X @ Vt[:3].T
    scale = []
    for a in range(3):                                 # Library.swift's Galaxy rule, applied here too
        mags = np.sort(np.abs(P[:, a]))
        p98 = max(mags[int((len(mags) - 1) * 0.98)], 1e-6)
        scale.append(float(p98))
        P[:, a] = np.clip(P[:, a] / p98, -1.3, 1.3)
    texts = [hf.decode([i]) for i in ids]
    scripts = [script_of(t) for t in texts]
    out = {"ids": ids,
           "points": [round(float(v), 4) for v in P.ravel()],
           "variance": [round(float(v), 6) for v in var[:3]],
           "script": scripts,
           "frequency": [int(counts[i]) for i in ids],
           "source": f"Cipher lines.tsv, {len(rows)} lines",
           "tokens": texts,
           "scale": [round(s, 6) for s in scale],
           "mean": [round(float(v), 6) for v in mean],
           "axes": [[round(float(v), 6) for v in Vt[i]] for i in range(3)],
           "note": "points = ((embedding - mean) . axes) / scale, clipped to +-1.3; embedding rows are the "
                   "dequantized 4-bit embed_tokens (float32)."}
    path = os.path.join(OUT, "qwen-catalogue.json")
    json.dump(out, open(path, "w"), ensure_ascii=False)
    tally = {s: scripts.count(s) for s in ["latin", "devanagari", "gujarati", "digit", "mark"]}
    log(f"catalogue: {len(ids)} ids, variance {var[:3].round(4).tolist()} (sum {var[:3].sum():.4f}),"
        f" scripts {tally}, wrote {path}")
    return out


# ------------------------------------------------------------------------------------------ reference

def rms(x, w):
    x = x.astype(np.float32)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(EPS))) * w


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def rope_tables(T):
    inv = (1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))).astype(np.float32)
    ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cos, sin):
    """Non-traditional (rotate-halves) RoPE over the last axis; x is [heads, T, 128]."""
    h = HEAD_DIM // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def reference(model, hf):
    """Explicit float32 numpy forward over dequantized weights, both prompts side by side."""
    np.seterr(over="ignore")
    log("reference: dequantizing embedding")
    E = deq(model.model.embed_tokens)                                  # [vocab, D] float32
    final_w = f32(model.model.norm.weight)
    seqs = {k: hf.encode(v, add_special_tokens=False) for k, v in REF_PROMPTS.items()}
    hs = {k: E[ids].copy() for k, ids in seqs.items()}
    trace = {k: {"layers": [hs[k]], "attention": {}} for k in seqs}
    for li, layer in enumerate(model.model.layers):
        a, m = layer.self_attn, layer.mlp
        Wq, Wk, Wv, Wo = deq(a.q_proj), deq(a.k_proj), deq(a.v_proj), deq(a.o_proj)
        Wg, Wu, Wd = deq(m.gate_proj), deq(m.up_proj), deq(m.down_proj)
        w_in, w_post = f32(layer.input_layernorm.weight), f32(layer.post_attention_layernorm.weight)
        qn, kn = f32(a.q_norm.weight), f32(a.k_norm.weight)
        for k, ids in seqs.items():
            T, h = len(ids), hs[k]
            cos, sin = rope_tables(T)
            xn = rms(h, w_in)
            q = rope(rms((xn @ Wq.T).reshape(T, HEADS, HEAD_DIM).transpose(1, 0, 2), qn), cos, sin)
            kk = rope(rms((xn @ Wk.T).reshape(T, KV_HEADS, HEAD_DIM).transpose(1, 0, 2), kn), cos, sin)
            v = (xn @ Wv.T).reshape(T, KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
            group = HEADS // KV_HEADS
            kk, v = np.repeat(kk, group, axis=0), np.repeat(v, group, axis=0)  # head h reads kv head h // 4
            s = (q @ kk.transpose(0, 2, 1)) / np.float32(np.sqrt(HEAD_DIM))
            s = np.where(np.tril(np.ones((T, T), bool))[None], s, np.float32(-np.inf))
            p = softmax(s.astype(np.float32))
            if li in REF_LAYERS:
                for hd in REF_HEADS:
                    trace[k]["attention"][f"L{li}H{hd}"] = [round(float(x), 7) for x in p[hd, -1]]
            o = (p @ v).transpose(1, 0, 2).reshape(T, HEADS * HEAD_DIM)
            h = h + o @ Wo.T
            hn = rms(h, w_post)
            g = hn @ Wg.T
            act = (g / (np.float32(1) + np.exp(-g))) * (hn @ Wu.T)
            if li == 17:
                last = act[-1]
                top = np.argsort(-np.abs(last))[:8]
                trace[k]["mlp17_top"] = [[int(i), round(float(last[i]), 5)] for i in top]
            h = h + act @ Wd.T
            hs[k] = h.astype(np.float32)
            trace[k]["layers"].append(hs[k])
        if li % 6 == 5:
            log(f"reference: layer {li + 1}/36")
    out = {"model": HF_NAME,
           "precision": "float32 activations; 4-bit weights dequantized to float32 (w*scale+bias, group 64)",
           "rope": "non-traditional (rotate halves), base 5e6, inv_freq[i] = base^(-2i/128), positions from 0",
           "attention_note": "softmax row of the last query over all keys; head h uses kv head h // 4; "
                             "scores = q.k / sqrt(128) after per-head q_norm/k_norm and RoPE",
           "mlp17_note": "top 8 by |value| of silu(gate_proj(h)) * up_proj(h), h = post_attention_layernorm output",
           "lens_note": "per layer output at the last position: final norm, tied readout, softmax -> [id, prob] of top",
           "records": {}}
    logits_all = {}
    for k, ids in seqs.items():
        tr = trace[k]
        logits = rms(tr["layers"][-1], final_w) @ E.T                    # [T, vocab]
        logits_all[k] = logits
        last = logits[-1]
        top = np.argsort(-last)[:10]
        lens = []
        for L in tr["layers"][1:]:
            pr = softmax(rms(L[-1], final_w) @ E.T)
            j = int(np.argmax(pr))
            lens.append([j, round(float(pr[j]), 6)])
        out["records"][k] = {
            "text": REF_PROMPTS[k],
            "ids": ids,
            "tokens": [hf.decode([i]) for i in ids],
            "top10": [[int(i), round(float(last[i]), 5)] for i in top],
            "top10_text": [hf.decode([int(i)]) for i in top],
            "logits_probe": [[i, round(float(last[i]), 5)] for i in PROBE_IDS + [int(top[0])]],
            "attention": tr["attention"],
            "resid_norm": [round(float(np.linalg.norm(L[-1])), 4) for L in tr["layers"]],
            "mlp17_top": tr["mlp17_top"],
            "lens_top1": lens,
            "lens_top1_text": [hf.decode([j]) for j, _ in lens],
        }
        log(f"reference[{k}]: {len(ids)} tokens, top5 {[hf.decode([int(i)]) for i in top[:5]]}")
    del E
    return out, logits_all, seqs


# ------------------------------------------------------------------------------------- manual mlx walk

def walk(model, x):
    """Batched manual forward: embedding, then each decoder layer by hand; returns every layer output."""
    h = model.model.embed_tokens(x)
    mask = "causal" if x.shape[1] > 1 else None
    outs = []
    for layer in model.model.layers:
        h = h + layer.self_attn(layer.input_layernorm(h), mask, None)
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        outs.append(h)
    return outs


def walk_logits(model, x):
    h = walk(model, x)[-1]
    return model.model.embed_tokens.as_linear(model.model.norm(h))


# ------------------------------------------------------------------------------------------- steering

def steering(model, rows, enc):
    rng = np.random.default_rng(SEED)
    ok = np.array([len(ids) >= 2 for ids in enc])                    # need a position after the first
    lit = np.array([r[0] == "lit" for r in rows])
    lang = np.array([r[1] for r in rows])
    groups = {}
    for l in LANGS:
        pool = np.nonzero(ok & lit & (lang == l))[0]
        groups["lang:" + l] = rng.choice(pool, min(PER_GROUP, len(pool)), replace=False)
    for fam in FAMILIES:
        pool = np.nonzero(ok & np.array([fam in r[3] for r in rows]))[0]
        groups["mood:" + fam] = rng.choice(pool, min(PER_GROUP, len(pool)), replace=False)
    groups["everyone"] = rng.choice(np.nonzero(ok)[0], EVERYONE, replace=False)
    names = list(groups)
    member = {}
    for gi, g in enumerate(names):
        for r in groups[g]:
            member.setdefault(int(r), []).append(gi)
    todo = sorted(member, key=lambda r: (min(len(enc[r]), MAX_TOKENS), r))
    log(f"steering: {len(todo)} unique lines; " + ", ".join(f"{g} {len(groups[g])}" for g in names))

    ck = os.path.join(OUT, ".steering-partial.npz")
    sums = np.zeros((len(names), LAYERS, D), np.float64)
    norm_sum = np.zeros(LAYERS, np.float64)
    done = set()
    if os.path.exists(ck):
        z = np.load(ck)
        if list(z["names"]) == names:
            sums, norm_sum, done = z["sums"], z["norm_sum"], set(z["done"].tolist())
            log(f"steering: resumed {len(done)} lines from checkpoint")
    todo = [r for r in todo if r not in done]

    for bi in range(0, len(todo), BATCH):
        batch = todo[bi:bi + BATCH]
        seqs = [enc[r][:MAX_TOKENS] for r in batch]
        T = max(len(s) for s in seqs)
        x = mx.array([s + [151643] * (T - len(s)) for s in seqs])     # right padding: causal mask keeps
        lens = mx.array([len(s) for s in seqs], dtype=mx.float32)     # real positions untouched
        pos = mx.arange(T)[None, :]
        w = ((pos >= 1) & (pos < lens[:, None])).astype(mx.float32)    # skip the first position + padding
        den = (lens - 1)[:, None]
        means, norms = [], []
        for h in walk(model, x):
            means.append((h * w[:, :, None]).sum(1) / den)                # [B, D]
            norms.append((mx.linalg.norm(h, axis=-1) * w).sum(1) / den[:, 0])
        means, norms = mx.stack(means, 1), mx.stack(norms, 1)          # [B, 36, D], [B, 36]
        mx.eval(means, norms)
        means, norms = np.array(means, np.float64), np.array(norms, np.float64)
        for j, r in enumerate(batch):
            for gi in member[r]:
                sums[gi] += means[j]
            norm_sum += norms[j]
            done.add(r)
        b = bi // BATCH
        if b % 10 == 9 or bi + BATCH >= len(todo):
            np.savez(ck, names=np.array(names), sums=sums, norm_sum=norm_sum, done=np.array(sorted(done)))
            log(f"steering: {len(done)}/{len(member)} lines (T={T})")

    counts = {g: len(groups[g]) for g in names}
    mean = {g: sums[i] / counts[g] for i, g in enumerate(names)}
    tensors = {}
    for l in LANGS:
        others = np.mean([mean["lang:" + o] for o in LANGS if o != l], axis=0)
        tensors["lang:" + l] = (mean["lang:" + l] - others).astype(np.float32)
    for fam in FAMILIES:
        tensors["mood:" + fam] = (mean["mood:" + fam] - mean["everyone"]).astype(np.float32)
    resid = (norm_sum / len(done)).tolist()
    meta = {"name": "Qwen steering directions",
            "residual_norm": [round(float(v), 4) for v in resid],
            "lines_per_group": PER_GROUP,
            "group_lines": counts,
            "method": "difference of mean residuals after each layer",
            "model": HF_NAME,
            "detail": f"raw text, no template; first {MAX_TOKENS} tokens; per line the mean over positions "
                      f"1.. (first excluded), then the mean over lines; language = its mean minus the mean of "
                      f"the other four; mood = its mean minus {EVERYONE} random lines; float32 activations; "
                      f"seed {SEED}"}
    path = os.path.join(OUT, "qwen-steering.loom")
    write_loom(path, meta, tensors, half=False)
    os.remove(ck)
    for k, v in tensors.items():
        nl = np.linalg.norm(v, axis=1)
        log(f"steering {k:13s} |dir| L0 {nl[0]:.2f}  L17 {nl[17]:.2f}  L35 {nl[35]:.2f}")
    log(f"steering: residual norm L0 {resid[0]:.1f}  L17 {resid[17]:.1f}  L35 {resid[35]:.1f}; wrote {path}")
    return counts


# ------------------------------------------------------------------------------------------------ main

def main():
    stages = sys.argv[1:] or ["catalogue", "reference", "steering"]
    os.makedirs(OUT, exist_ok=True)
    mdir = model_dir()
    log("loading", mdir)
    model, tok = load(mdir)
    model.set_dtype(mx.float32)            # widen bf16 scales/biases/norms (exact): float32 activations
    hf = tok._tokenizer
    timings, summary = {}, {}

    rows = enc = None
    if "catalogue" in stages or "steering" in stages:
        rows = read_corpus()
        t = time.time()
        enc = encode_all(hf, [r[4] for r in rows])
        log(f"tokenized {len(rows)} lines, {sum(map(len, enc)):,} tokens in {time.time() - t:.1f}s")

    if "catalogue" in stages:
        t = time.time()
        cat = catalogue(model, hf, rows, enc)
        timings["catalogue"] = time.time() - t
        summary["pca_variance"] = cat["variance"]

    if "reference" in stages:
        t = time.time()
        ref, logits_np, seqs = reference(model, hf)
        # checks: the explicit numpy forward and the batched mlx walk against model(x)
        checks = {}
        for k, ids in seqs.items():
            x = mx.array([ids])
            full = f32(model(x)[0])
            manual = f32(walk_logits(model, x)[0])
            checks[k] = {"walk_vs_model": float(np.abs(manual - full).max()),
                         "numpy_ref_vs_model": float(np.abs(logits_np[k] - full).max()),
                         "logit_range": [float(full.min()), float(full.max())]}
        # padding check: a line inside a right-padded batch gives the same last-layer output as alone
        a, b = seqs["raw"], seqs["chat"]
        padded = walk(model, mx.array([a + [151643] * (len(b) - len(a)), b]))[-1]
        alone = walk(model, mx.array([a]))[-1]
        checks["padding_max_abs_diff_last_layer"] = float(mx.abs(padded[0, :len(a)] - alone[0]).max())
        # context only: the stock bf16 model against the float32 reference
        bf, _ = load(mdir)
        checks["bf16_model_vs_f32_ref"] = {k: float(np.abs(f32(bf(mx.array([ids]))[0]) - logits_np[k]).max())
                                           for k, ids in seqs.items()}
        del bf
        ref["checks"] = checks
        path = os.path.join(OUT, "qwen-reference.json")
        json.dump(ref, open(path, "w"), ensure_ascii=False, indent=1)
        timings["reference"] = time.time() - t
        summary["checks"] = checks
        summary["top5"] = {k: r["top10_text"][:5] for k, r in ref["records"].items()}
        log("reference checks", json.dumps(checks))
        log("wrote", path)

    if "steering" in stages:
        t = time.time()
        summary["group_lines"] = steering(model, rows, enc)
        timings["steering"] = time.time() - t

    summary["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    summary["total_s"] = round(time.time() - T0, 1)
    summary["sizes"] = {f: os.path.getsize(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT))
                        if not f.startswith(".")}
    log("SUMMARY " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
