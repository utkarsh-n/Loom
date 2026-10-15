#!/usr/bin/env python3
"""Fine-tune the five mood adapters (LoRA, rank 8) on the base model, one per family of Cipher mood tags.

  adapters.py <run name> [--steps 1500]

Each family's training set is every line from a work Cipher tagged with one of its moods
(prepare.py: dread, tender, wonder, rousing, wry). The base stays frozen; only A and B train.
Writes <run>/adapter-<family>.loom with the before/after losses and samples in its header.
"""
import argparse, json, os, time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from bpe import Tokenizer
from loomio import DATA, SETS, VOICES, build_stream, voice_rows, write_loom
from mlx.utils import tree_flatten, tree_unflatten
from model import GPT, Config, LoRA
from train import generate, sample_batch, select_rows

TITLES = {v: v.capitalize() for v in VOICES}
PROMPTS = ["The night", "She looked at him", "I am", "Forrest Gump ("]


def load_base(run):
    meta = json.load(open(os.path.join(DATA, "runs", run, "run.json")))
    model = GPT(Config(**meta["config"]))
    model.load_weights(os.path.join(DATA, "runs", run, "weights.safetensors"))
    model.freeze()
    return model


def gen(model, tok, lora, prompt, seed=3):
    """generate() with an adapter mixed in."""
    mx.random.seed(seed)
    ids = [1] + tok.encode(prompt)
    for _ in range(36):
        logits = model(mx.array([ids[-model.c.ctx:]]), lora)[0, -1] / 0.8
        nxt = mx.random.categorical(logits).item()
        ids.append(nxt)
        if nxt == 1:
            break
    return tok.decode(ids[1:]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated families to train")
    a = ap.parse_args()
    tok = Tokenizer.load()
    s = build_stream(tok)
    base = load_base(a.run)
    c = base.c
    gen_ids, gen_starts = select_rows(s, s["val"])

    def loss_fn(lora_mod, x, y):
        return nn.losses.cross_entropy(base(x, lora_mod.triples()), y, reduction="mean")

    def eval_loss(lora_mod, ids, starts, n=30):
        rng = np.random.default_rng(9)
        tot = 0.0
        for _ in range(n):
            x, y = sample_batch(ids, starts, 64, c.ctx, rng)
            tot += (loss_fn(lora_mod, x, y) if lora_mod else nn.losses.cross_entropy(base(x), y, reduction="mean")).item()
        return tot / n

    only = [f for f in a.only.split(",") if f] or VOICES
    for bit, fam in enumerate(VOICES):
        if fam not in only:
            continue
        mask = voice_rows(fam)
        held = np.zeros_like(mask)
        held[np.nonzero(mask)[0][::10]] = True
        tr_ids, tr_starts = select_rows(s, mask & ~held)
        va_ids, va_starts = select_rows(s, held)
        mx.random.seed(100 + bit)
        lora = LoRA(c, rank=a.rank, alpha=2.0 * a.rank)
        before = eval_loss(None, va_ids, va_starts)
        general_before = eval_loss(None, gen_ids, gen_starts)
        sched = optim.cosine_decay(a.lr, a.steps, a.lr * 0.1)
        opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
        vg = nn.value_and_grad(lora, loss_fn)
        state = [lora.state, opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(x, y):
            loss, g = vg(lora, x, y)
            opt.update(lora, g)
            return loss

        rng = np.random.default_rng(bit)
        t0 = time.time()
        # Early stopping: keep the adapter from the step where held-out loss was lowest.
        best, best_step, curve = before, 0, [[0, round(before, 4)]]
        best_params = {k: mx.array(v) for k, v in tree_flatten(lora.parameters())}
        for it in range(1, a.steps + 1):
            x, y = sample_batch(tr_ids, tr_starts, 32, c.ctx, rng)
            step(x, y)
            mx.eval(state)
            if it % 10 == 0:
                vl = eval_loss(lora, va_ids, va_starts, n=12)
                curve.append([it, round(vl, 4)])
                if vl < best:
                    best, best_step = vl, it
                    best_params = {k: mx.array(v) for k, v in tree_flatten(lora.parameters())}
        lora.update(tree_unflatten(list(best_params.items())))
        after = eval_loss(lora, va_ids, va_starts)
        general_after = eval_loss(lora, gen_ids, gen_starts)
        samples = {p: {"base": gen(base, tok, None, p), "adapter": gen(base, tok, lora.triples(), p),
                       "double": gen(base, tok, lora.triples(2.0), p)} for p in PROMPTS}
        print(f"{fam:8s} {len(tr_ids):>9,} tokens  best step {best_step}  family val {before:.3f} → {after:.3f}  "
              f"general {general_before:.3f} → {general_after:.3f}  {time.time()-t0:.0f}s")
        for p, v in samples.items():
            print("   ", repr(v["adapter"])[:110])
        tensors = {}
        for i, layer in enumerate(lora.layers):
            for k, v in layer.items():
                tensors[f"b{i}.{k}"] = np.array(v)
        meta = {"name": TITLES[fam], "voice": fam, "works": SETS[fam] or ["Cipher culture-works fact cards"], "lines": int(mask.sum()), "rank": a.rank, "alpha": 2.0 * a.rank,
                "train_tokens": int(len(tr_ids)), "steps": a.steps, "lr": a.lr,
                "best_step": best_step, "curve": curve, "family_val_before": round(before, 4), "family_val_after": round(after, 4),
                "general_val_before": round(general_before, 4), "general_val_after": round(general_after, 4),
                "samples": samples, "parameters": int(sum(v.size for v in tensors.values()))}
        write_loom(os.path.join(DATA, "runs", a.run, f"adapter-{fam}.loom"), meta, tensors, half=True)


if __name__ == "__main__":
    main()
