#!/usr/bin/env python3
"""Measurements of the finished base model that the app ships alongside it.

  extras.py <run name>

Writes, into <run>/:
  steering.loom   difference-of-means directions on every floor: each language minus the other four,
                  each mood family minus the corpus. Meta records the typical residual norm per floor.
  neurons.json    for every SwiGLU neuron, the three corpus moments that light it up most.
  galaxy.json     PCA of the card catalogue (3 axes, variance explained) and each token's script.
All measured with the float64 reference forward pass, the same arithmetic the app runs.
"""
import json, os, sys, zlib

import numpy as np

from bpe import Tokenizer
from loomio import DATA, FAMILIES, LANGS, build_stream, write_loom
from ref import Weights, forward


def script_of(t):
    for ch in t:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: return "devanagari"
        if 0x0A80 <= o <= 0x0AFF: return "gujarati"
        if ch.isalpha(): return "latin"
        if ch.isdigit(): return "digit"
    return "mark"


def main():
    run = sys.argv[1]
    W = Weights(os.path.join(DATA, "runs", run, "loom-base.loom"))
    tok = Tokenizer.load()
    s = build_stream(tok)
    ids, starts = s["ids"], s["starts"]
    n_rows = len(starts) - 1
    rng = np.random.default_rng(17)

    def line(r):
        return ids[starts[r]:starts[r + 1]].tolist()[:W.ctx]

    # ---- residual means per floor
    def means(rows):
        acc = np.zeros((W.layers, W.d))
        n = 0
        for r in rows:
            seq = line(r)
            if len(seq) < 4:
                continue
            f = forward(W, seq)
            for l, L in enumerate(f["layers"]):
                acc[l] += L["out"][1:].sum(0)
            n += len(seq) - 1
        return acc / max(n, 1)

    lit = s["card"] == 0
    groups = {}
    for li, lang in enumerate(LANGS):
        rows = np.nonzero(lit & (s["lang"] == li))[0]
        groups["lang:" + lang] = rng.choice(rows, min(350, len(rows)), replace=False)
    for bit, fam in enumerate(FAMILIES):
        rows = np.nonzero((s["family"] & (1 << bit)) > 0)[0]
        groups["mood:" + fam] = rng.choice(rows, min(350, len(rows)), replace=False)
    group_means = {k: means(v) for k, v in groups.items()}
    everyone = means(rng.choice(np.arange(n_rows), 900, replace=False))
    tensors, norms = {}, []
    for k, m in group_means.items():
        if k.startswith("lang:"):
            others = np.mean([v for kk, v in group_means.items() if kk.startswith("lang:") and kk != k], axis=0)
            vec = m - others
        else:
            vec = m - everyone
        tensors[k] = vec.astype(np.float32)                 # layers × d
        print(k, " ".join(f"{np.linalg.norm(vec[l]):.2f}" for l in range(W.layers)))
    probe = [line(r) for r in rng.choice(np.arange(n_rows), 200, replace=False)]
    for l in range(W.layers):
        vals = [np.linalg.norm(forward(W, p)["layers"][l]["out"][1:], axis=1).mean() for p in probe[:60] if len(p) > 3]
        norms.append(round(float(np.mean(vals)), 4))
    write_loom(os.path.join(DATA, "runs", run, "steering.loom"),
               {"name": "Steering directions", "residual_norm": norms, "lines_per_group": 350,
                "method": "difference of mean residuals after each floor"}, tensors)

    # ---- neuron cards
    top = np.full((W.layers, W.ff, 3), -np.inf)
    where = np.zeros((W.layers, W.ff, 3, 2), dtype=np.int64)
    for r in rng.choice(np.arange(n_rows), 2500, replace=False):
        seq = line(r)
        if len(seq) < 3:
            continue
        f = forward(W, seq)
        for l, L in enumerate(f["layers"]):
            act = L["act"]                                   # T × ff
            best_pos = act.argmax(0)
            best = act[best_pos, np.arange(W.ff)]
            for n in np.nonzero(best > top[l, :, 2])[0]:
                if r in where[l, n, :, 0][np.isfinite(top[l, n])]:
                    continue
                slot = 2
                top[l, n, 2], where[l, n, 2] = best[n], (r, best_pos[n])
                while slot > 0 and top[l, n, slot] > top[l, n, slot - 1]:
                    top[l, n, [slot - 1, slot]] = top[l, n, [slot, slot - 1]]
                    where[l, n, [slot - 1, slot]] = where[l, n, [slot, slot - 1]]
                    slot -= 1
    cards = []
    for l in range(W.layers):
        layer = []
        for n in range(W.ff):
            moments = []
            for k in range(3):
                if not np.isfinite(top[l, n, k]):
                    continue
                r, p = where[l, n, k]
                seq = line(int(r))
                moments.append([round(float(top[l, n, k]), 3), tok.decode(seq[max(1, p - 7):p]).replace("\n", " "),
                                tok.vocab[seq[p]], tok.decode(seq[p + 1:p + 4]).replace("\n", " ")])
            layer.append(moments)
        cards.append(layer)
    json.dump({"layers": cards, "lines": 2500}, open(os.path.join(DATA, "runs", run, "neurons.json"), "w"), ensure_ascii=False)

    # ---- galaxy
    E = W["wte"]
    X = E - E.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    P = X @ Vt[:3].T
    P /= np.abs(P).max()
    json.dump({"variance": [round(float(v), 4) for v in var[:3]],
               "points": [round(float(v), 4) for v in P.ravel()],
               "script": [script_of(t) for t in tok.vocab],
               "axes": [[round(float(v), 5) for v in Vt[i]] for i in range(3)],
               "mean": [round(float(v), 5) for v in E.mean(0)]},
              open(os.path.join(DATA, "runs", run, "galaxy.json"), "w"))
    print("galaxy variance", var[:3], "sum", var[:3].sum())


if __name__ == "__main__":
    main()
