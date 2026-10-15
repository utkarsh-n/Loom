#!/usr/bin/env python3
"""Build Loom's training corpus from the data Cipher already ships.

Sources (read-only):
  Cipher/Cipher/Resources/culture-lines.tsv   public-domain lines: key, year, lang, work, author, line
  Cipher/Cipher/Resources/culture-works.tsv   one work per row (16 columns, see below)

Writes to $LOOM_DATA (default ~/Developer/loom-data):
  lines.tsv     slice, lang, work, moods, text      one cleaned, de-duplicated line per row
  stats.txt     counts per language, slice and mood family

Slices:
  lit    a line from a public-domain book, play or poem (all five languages)
  card   a fact card written from a culture-works row: title, year, kind, creator, meaning, moods

Mood families come from Cipher's own culture tags (column 14 of culture-works.tsv), folded into five:
dread, tender, wonder, rousing, wry. They are the training sets for the five style adapters.
"""
import collections, os, re, sys, unicodedata

CIPHER = os.environ.get("CIPHER_RES", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Cipher-public/Cipher/Resources")))
OUT = os.environ.get("LOOM_DATA", os.path.expanduser("~/Developer/loom-data"))

FAMILIES = {
    "dread": "uncanny dread menacing haunted grim unsettling harrowing dangerous suspicious brooding tense suspenseful gothic eerie ominous dark",
    "tender": "tender yearning warm lyrical tearful mournful bittersweet passionate sensual lonesome romantic wistful",
    "wonder": "wondrous awestruck innocent bright light exuberant free whimsical magical dreamy",
    "rousing": "rousing stately grand sweeping heightened explosive adrenalized rugged epic triumphant",
    "wry": "biting wry ironic roguish playful satirical witty sardonic absurd",
}
MOOD_TO_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms.split()}

# culture-works.tsv columns (1-based): 1 qid, 2 kind, 3 country, 4 year, 5 score, 6 title, 7 alt title,
# 8 creator, 9 description, 10 honours, 11 themes, 12 motifs, 13 meaning, 14 moods, 15 siblings, 16 flags
def parse_tags(field):
    out = []
    for part in field.split("|"):
        if not part:
            continue
        name = part.rsplit(":", 1)[0] if re.search(r":\d", part) else part
        out.append(name.strip())
    return out


QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', " ": " ", " ": " ", " ": " "})


def clean(text):
    text = text.replace("�", "")
    text = unicodedata.normalize("NFC", text).translate(QUOTES)
    text = re.sub(r"^\s*\d+:\d+\s+", "", text)            # scripture verse numbers
    text = re.sub(r"_([^_]+)_", r"\1", text)               # Gutenberg italics
    text = text.replace("_", " ")
    text = re.sub(r"\[\d+\]", "", text)                     # footnote marks
    text = re.sub(r"\s*--\s*", " — ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # The concordance cut some lines mid-character; drop a dangling combining mark at the end.
    while text and unicodedata.category(text[-1]).startswith("M") and len(text) > 1 and unicodedata.category(text[-2]) in ("Mn", "Mc"):
        text = text[:-1]
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    works = {}
    cards = []
    with open(os.path.join(CIPHER, "culture-works.tsv"), encoding="utf-8") as f:
        for row in f:
            c = row.rstrip("\n").split("\t")
            if len(c) < 16:
                continue
            title, kind, year, creator = c[5], c[1], c[3], c[7]
            moods = parse_tags(c[13])
            fams = sorted({MOOD_TO_FAMILY[m] for m in moods if m in MOOD_TO_FAMILY})
            works.setdefault(title, (fams, moods))
            score = int(c[4] or 0)
            meaning = c[12].strip()
            if score >= 55 and title:
                head = f"{title} ({year} {kind}" if year else f"{title} ({kind}"
                if creator:
                    head += f", {creator.split(',')[0].strip()}"
                head += ")"
                body = meaning or c[8].strip()
                tail = ", ".join(moods[:3])
                card = head + (": " + body if body else "") + ("." if body and not body.endswith(".") else "")
                if tail:
                    card += " " + tail[0].upper() + tail[1:] + "."
                cards.append((c[2], title, "|".join(fams), clean(card)))

    seen = set()
    rows = []
    with open(os.path.join(CIPHER, "culture-lines.tsv"), encoding="utf-8") as f:
        for row in f:
            c = row.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            lang, work, text = c[2], c[3], clean(c[5])
            if len(text) < 12 or (lang, text) in seen:
                continue
            seen.add((lang, text))
            fams = works.get(work, ([], []))[0]
            rows.append(("lit", lang, work, "|".join(fams), text))

    for country, title, fams, card in cards:
        if card in seen:
            continue
        seen.add(card)
        rows.append(("card", "en", title, fams, card))

    # Group by work so a training window holds one voice at a time.
    rows.sort(key=lambda r: (r[0], r[2], r[1]))
    with open(os.path.join(OUT, "lines.tsv"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(x.replace("\t", " ") for x in r) + "\n")

    by_lang = collections.Counter((r[0], r[1]) for r in rows)
    by_fam = collections.Counter(f for r in rows for f in r[3].split("|") if f)
    chars = sum(len(r[4]) for r in rows)
    with open(os.path.join(OUT, "stats.txt"), "w") as f:
        print(f"rows {len(rows)}  characters {chars}", file=f)
        for k, v in sorted(by_lang.items()):
            print(f"  {k[0]:5} {k[1]}  {v}", file=f)
        for k, v in by_fam.most_common():
            print(f"  family {k:8} {v}", file=f)
    print(open(os.path.join(OUT, "stats.txt")).read())


if __name__ == "__main__":
    main()
