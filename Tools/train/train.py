#!/usr/bin/env python3
"""Train Loom's base model on the Cipher corpus with MLX, then export Loom/Resources/loom-base.loom.

  train.py [--steps 8000] [--batch 128] [--lr 2e-3] [--d 128 --layers 4 --heads 4 --ctx 64]

Checkpoints and logs go to $LOOM_DATA/runs/<name>/. The exported file carries the config,
the loss curve and the corpus counts in its JSON header so the app can show them.
"""
import argparse, json, math, os, time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from bpe import Tokenizer
from loomio import DATA, adapter_rows, build_stream, write_loom
from model import GPT, Config, MATRICES

APP_RES = os.path.join(os.path.dirname(__file__), "..", "..", "Loom", "Resources")


def select_rows(s, keep):
    """Concatenate the chosen rows into their own stream; return ids and row starts."""
    starts, ids = s["starts"], s["ids"]
    pieces, new_starts, n = [], [], 0
    for r in np.nonzero(keep)[0]:
        seg = ids[starts[r]:starts[r + 1]]
        new_starts.append(n)
        pieces.append(seg)
        n += len(seg)
    return np.concatenate(pieces), np.array(new_starts)


def sample_batch(ids, starts, B, T, rng, aligned=0.7):
    n_al = int(B * aligned)
    ok = starts[starts < len(ids) - T - 1]
    pos = np.concatenate([rng.choice(ok, n_al), rng.integers(0, len(ids) - T - 1, B - n_al)])
    X = np.stack([ids[p:p + T + 1] for p in pos]).astype(np.int32)
    return mx.array(X[:, :-1]), mx.array(X[:, 1:])


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def evaluate(model, ids, starts, T, n=40, seed=7):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n):
        x, y = sample_batch(ids, starts, 64, T, rng)
        total += loss_fn(model, x, y).item()
    return total / n


def generate(model, tok, prompt, n=40, temp=0.8, seed=0, tag=None):
    mx.random.seed(seed)
    ids = [1] + ([tok.index[tag]] if tag else []) + tok.encode(prompt)
    for _ in range(n):
        x = mx.array([ids[-model.c.ctx:]])
        logits = model(x)[0, -1] / temp
        nxt = mx.random.categorical(logits).item()
        ids.append(nxt)
        if nxt == 1:
            break
    return tok.decode(ids[1:])


def flat_tensors(model):
    p = dict(tree_flatten(model.parameters()))
    t = {"wte": p["wte"], "gf": p["gf"]}
    for i in range(model.c.layers):
        for n in ("g1", "g2") + MATRICES:
            t[f"b{i}.{n}"] = p[f"blocks.{i}.{n}"]
    return {k: np.array(v) for k, v in t.items()}


def export(model, path, meta, half=False):
    write_loom(path, meta, flat_tensors(model), half=half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--ff", type=int, default=0, help="SwiGLU hidden size (default 3·d)")
    ap.add_argument("--ages", default="40,250,1500", help="steps at which to keep a training-age checkpoint")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    tok = Tokenizer.load()
    s = build_stream(tok)
    held = adapter_rows()            # the adapters' books never reach the base
    tr_ids, tr_starts = select_rows(s, ~s["val"] & ~held)
    va_ids, va_starts = select_rows(s, s["val"] & ~held)
    print(f"held out for adapters: {int(held.sum()):,} lines")
    print(f"train tokens {len(tr_ids):,}  validation tokens {len(va_ids):,}")

    c = Config(vocab=len(tok.vocab), ctx=a.ctx, d=a.d, heads=a.heads, layers=a.layers, ff=a.ff or 3 * a.d)
    mx.random.seed(a.seed)
    model = GPT(c)
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"parameters {n_params:,}")

    warm = min(300, a.steps // 10)
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-5, a.lr, warm), optim.cosine_decay(a.lr, a.steps - warm, a.lr * 0.05)], [warm])
    opt = optim.AdamW(learning_rate=sched, betas=[0.9, 0.95], weight_decay=0.05)
    vg = nn.value_and_grad(model, loss_fn)

    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = vg(model, x, y)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    run = os.path.join(DATA, "runs", a.name)
    os.makedirs(run, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ages = {int(x) for x in a.ages.split(",") if x}
    samples = {}
    probe = [("The old man", None), ("Il était une fois", None), ("El caballero", None), ("मैं", None),
             ("Forrest Gump (", "⟨card⟩"), ("Love is", None), ("The night", "⟨es⟩")]
    curve, t0, avg = [], time.time(), None
    for it in range(1, a.steps + 1):
        x, y = sample_batch(tr_ids, tr_starts, a.batch, c.ctx, rng)
        loss = step(x, y)
        mx.eval(state)
        avg = loss.item() if avg is None else 0.98 * avg + 0.02 * loss.item()
        if it % 250 == 0 or it == a.steps:
            vl = evaluate(model, va_ids, va_starts, c.ctx)
            curve.append([it, round(avg, 4), round(vl, 4)])
            print(f"{it:6d}  train {avg:.3f}  val {vl:.3f}  {time.time()-t0:.0f}s", flush=True)
        if it in ages:
            vl = evaluate(model, va_ids, va_starts, c.ctx)
            samples[it] = [generate(model, tok, p, tag=t) for p, t in probe]
            export(model, os.path.join(run, f"loom-age-{it}.loom"),
                   {"name": f"Loom at step {it}", "config": c.dict(), "steps": it, "val_loss": round(vl, 4)}, half=True)
            print(f"  age checkpoint {it}: val {vl:.3f}", samples[it][0][:60].replace("\n", " "), flush=True)
    model.save_weights(os.path.join(run, "weights.safetensors"))
    json.dump({"config": c.dict(), "curve": curve}, open(os.path.join(run, "run.json"), "w"))

    samples[a.steps] = [generate(model, tok, p, tag=t) for p, t in probe]
    for line in samples[a.steps]:
        print(repr(line))
    json.dump(samples, open(os.path.join(run, "samples.json"), "w"), ensure_ascii=False, indent=1)

    meta = {
        "name": "Loom base", "config": c.dict(), "parameters": n_params, "steps": a.steps,
        "batch_tokens": a.batch * c.ctx, "train_tokens": int(len(tr_ids)), "val_tokens": int(len(va_ids)),
        "curve": curve, "trained": time.strftime("%Y-%m-%d"), "ages": sorted(ages) + [a.steps], "corpus": "Cipher culture-lines + culture-works cards", "held_out_lines": int(held.sum()),
    }
    export(model, os.path.join(run, "loom-base.loom"), meta)
    print("exported", os.path.join(run, "loom-base.loom"))


if __name__ == "__main__":
    main()
