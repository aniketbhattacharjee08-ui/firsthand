"""data_v4: data_v3 pairs with TERSE notes.

Why (bench r3, 2026-09-10): the author's facts for explainer-2 are full
sentences and the adapter reproduced them verbatim, so every gate-passing
candidate was the same stitched paragraph and GPTZero read it as AI at
1.000. The training notes were clause windows lifted from the target, which
taught copying. Here each note is the particular itself with at most two
words of context on either side, so the model must compose the sentence.

    .venv/bin/python .hip7b/build_data_v4.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from build_data_v2 import missing_particulars  # noqa: E402
from humanizer.humanize.llm import _FREEFORM_NO_NOTES  # noqa: E402
from humanizer.humanize.pipeline import new_specifics  # noqa: E402
from humanizer.text import split_sentences  # noqa: E402

MAX_ITEMS = 8
CONTEXT = 2


def terse(sentence: str, needle: str, context: int = CONTEXT) -> str:
    words = sentence.split()
    idx = [i for i, w in enumerate(words) if needle in w or (" " in needle and needle in " ".join(words[i : i + len(needle.split())]))]
    if not idx:
        return needle
    i = idx[0]
    span = len(needle.split())
    lo, hi = max(0, i - context), min(len(words), i + span + context)
    return " ".join(words[lo:hi]).strip(" ,;:.")


def terse_notes(draft: str, human: str, rng: random.Random):
    missing = missing_particulars(draft, human)
    if not missing:
        return _FREEFORM_NO_NOTES
    sentences = split_sentences(human)
    items = []
    for needle in missing:
        for s in sentences:
            if needle in s:
                t = terse(s, needle)
                if t not in items:
                    items.append(t)
                break
        else:
            items.append(needle)
    if len(items) > MAX_ITEMS:
        return None
    rng.shuffle(items)
    return "\n".join("- " + it for it in items)


def main() -> int:
    rng = random.Random(4)
    out = HERE / "data_v4"
    out.mkdir(exist_ok=True)
    for split in ("train", "valid"):
        kept = dropped = leaks = 0
        with (HERE / "data_v3" / f"{split}.jsonl").open() as f, (out / f"{split}.jsonl").open("w") as g:
            for line in f:
                r = json.loads(line)
                prompt, completion = r["prompt"], r["completion"]
                head, _, tail = prompt.rpartition("DRAFT:\n")
                draft = tail.split("\n\nNOTES:\n", 1)[0]
                human = re.sub(r"\n\nDRAFT:\n$", "", completion).strip()
                notes = terse_notes(draft, human, rng)
                if notes is None:
                    dropped += 1
                    continue
                if new_specifics(draft, human, facts=notes if notes != _FREEFORM_NO_NOTES else ""):
                    leaks += 1
                    dropped += 1
                    continue
                r["prompt"] = head + "DRAFT:\n" + draft + "\n\nNOTES:\n" + notes + "\n\nHUMAN:\n"
                r["completion"] = human + "\n\nDRAFT:\n"
                g.write(json.dumps(r) + "\n")
                kept += 1
        print("%s: kept %d, dropped %d (unlicensed after terse notes: %d)" % (split, kept, dropped, leaks), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
