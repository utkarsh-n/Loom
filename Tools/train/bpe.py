#!/usr/bin/env python3
"""Loom's tokenizer: character-level byte-pair encoding, trained on lines.tsv.

The Swift engine (Loom/Engine/Tokenizer.swift) reimplements `pieces` and `encode` exactly;
Tools/train/fixtures.py writes cases that the parity tests compare.

Pre-tokenising splits text into pieces that merges never cross:
  - "\n" is its own piece (it separates lines and starts every prompt);
  - a piece is an optional single leading space plus a run of one character class:
    L = letters, combining marks and the apostrophe; N = numbers; P = everything else;
  - a space that is not followed by L, N or P is a piece of its own.
Characters seen fewer than MIN_CHAR times become "<?>" (token 0). Tokens 2–7 are the control tokens.

Usage: bpe.py [vocab_size]     writes $LOOM_DATA/tokenizer.json
"""
import collections, json, os, sys, time, unicodedata

OUT = os.environ.get("LOOM_DATA", os.path.expanduser("~/Developer/loom-data"))
MIN_CHAR = 30
UNK, NL = "<?>", "\n"
# Control tokens ("cartridges"): half the training lines start with their language tag, fact cards with
# ⟨card⟩. The app inserts one after the opening "\n" when a cartridge is loaded; plain text never encodes to them.
SPECIALS = ["⟨en⟩", "⟨es⟩", "⟨fr⟩", "⟨hi⟩", "⟨gu⟩", "⟨card⟩"]


def char_class(c):
    if c == "\n":
        return "NL"
    if c == " ":
        return "SP"
    if c == "'":
        return "L"
    cat = unicodedata.category(c)
    if cat[0] in "LM":
        return "L"
    if cat[0] == "N":
        return "N"
    if cat[0] == "Z" or c in "\t\r\f\v":
        return "SP"
    return "P"


def pieces(text):
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            out.append(c); i += 1; continue
        lead = ""
        if char_class(c) == "SP":
            if i + 1 < n and char_class(text[i + 1]) in ("L", "N", "P"):
                lead = " "; i += 1
            else:
                out.append(" "); i += 1; continue
        start, k = i, char_class(text[i])
        while i < n and char_class(text[i]) == k:
            i += 1
        out.append(lead + text[start:i])
    return out


class Tokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.index = {t: i for i, t in enumerate(vocab)}
        self.ranks = {tuple(m): r for r, m in enumerate(merges)}
        self.cache = {}

    @classmethod
    def load(cls, path=None):
        d = json.load(open(path or os.path.join(OUT, "tokenizer.json"), encoding="utf-8"))
        return cls(d["vocab"], d["merges"])

    def bpe(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        syms = [c if c in self.index else UNK for c in piece]
        while len(syms) > 1:
            best, at = None, -1
            for j in range(len(syms) - 1):
                r = self.ranks.get((syms[j], syms[j + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, j
            if best is None:
                break
            a, b = syms[at], syms[at + 1]
            merged, j = [], 0
            while j < len(syms):
                if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                    merged.append(a + b); j += 2
                else:
                    merged.append(syms[j]); j += 1
            syms = merged
        ids = [self.index[s] for s in syms]
        self.cache[piece] = ids
        return ids

    def encode(self, text):
        return [i for p in pieces(text) for i in self.bpe(p)]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def train(vocab_size):
    t0 = time.time()
    chars = collections.Counter()
    words = collections.Counter()
    with open(os.path.join(OUT, "lines.tsv"), encoding="utf-8") as f:
        for row in f:
            text = row.rstrip("\n").split("\t")[4]
            chars.update(text)
            words.update(pieces(text))
    base = [c for c, n in chars.most_common() if n >= MIN_CHAR and c != "\n"]
    vocab = [UNK, NL] + SPECIALS + base
    known = set(vocab)
    print(f"{len(words)} distinct pieces, {len(base)} base characters ({time.time()-t0:.1f}s)")

    wlist = [[c if c in known else UNK for c in w] for w in words]
    counts = list(words.values())
    pair_counts = collections.Counter()
    where = collections.defaultdict(set)
    for wi, syms in enumerate(wlist):
        for a, b in zip(syms, syms[1:]):
            pair_counts[(a, b)] += counts[wi]
            where[(a, b)].add(wi)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if pair_counts[best] < 2:
            break
        a, b = best
        new = a + b
        merges.append([a, b])
        vocab.append(new)
        for wi in list(where[best]):
            syms, c = wlist[wi], counts[wi]
            for x, y in zip(syms, syms[1:]):
                pair_counts[(x, y)] -= c
                if pair_counts[(x, y)] <= 0:
                    del pair_counts[(x, y)]
                where[(x, y)].discard(wi)
            merged, j = [], 0
            while j < len(syms):
                if j < len(syms) - 1 and syms[j] == a and syms[j + 1] == b:
                    merged.append(new); j += 2
                else:
                    merged.append(syms[j]); j += 1
            wlist[wi] = merged
            for x, y in zip(merged, merged[1:]):
                pair_counts[(x, y)] += c
                where[(x, y)].add(wi)
        if len(merges) % 500 == 0:
            print(f"  {len(merges)} merges, last {new!r} ({time.time()-t0:.0f}s)")
    with open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "merges": merges, "min_char": MIN_CHAR}, f, ensure_ascii=False)
    print(f"vocab {len(vocab)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 4096)
