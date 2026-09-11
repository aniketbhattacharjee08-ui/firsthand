"""Shared code for the .hip7b adapter bench: paths, bench texts, metrics, tables.

Nothing in this module imports `humanizer` at import time. The pipeline's
`llm.py` reads HUMANIZER_BASE_MODEL and friends when it is *imported*, so the
CLIs set the environment first and only then call the functions below, which
import what they need lazily. Nothing here loads a model or calls a network.

Python 3.9: no `match`, no `X | Y` annotations.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
RESULTS_DIR = HERE / "results"
KRIUKOW = ROOT / ".kriukow"
DEFAULT_TEXTS = KRIUKOW / "gptzero_texts.json"
DEFAULT_FACTS = KRIUKOW / "gptzero_facts.json"
CACHE_DIR = ROOT / "data" / "cache" / "gptzero"

#: Mirrors humanizer.detectors.gptzero; duplicated so `--help` and the
#: baseline table never import the package.
COST_PER_WORD_USD = 0.00046
MIN_CHARS = 250
HUMAN_THRESHOLD = 0.5


def ensure_src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


# ------------------------------------------------------------- bench texts


def guess_kind(name: str, kind: Optional[str] = None) -> str:
    if kind in ("ai", "human"):
        return kind
    return "ai" if name.lower().startswith("ai") else "human"


def load_bench(
    texts_path: Path = DEFAULT_TEXTS,
    facts_path: Optional[Path] = DEFAULT_FACTS,
    ai_only: bool = False,
    limit: int = 0,
) -> List[Dict[str, str]]:
    """The 14 bench paragraphs (9 AI with author facts, 5 human PMC controls)."""
    items = json.loads(Path(texts_path).read_text())
    facts: Dict[str, str] = {}
    if facts_path and Path(facts_path).exists():
        facts = json.loads(Path(facts_path).read_text())
    out: List[Dict[str, str]] = []
    for it in items:
        kind = guess_kind(it["name"], it.get("kind"))
        if ai_only and kind != "ai":
            continue
        out.append({
            "name": it["name"],
            "text": it["text"],
            "kind": kind,
            "facts": facts.get(it["name"], "") or "",
        })
    if limit:
        out = out[:limit]
    return out


# ------------------------------------------------------------------ metrics


_CONTRACTION_RE = re.compile(
    r"\b[A-Za-z]+(?:n['’]t|['’](?:re|ve|ll|d|m))\b"
    r"|\b(?:it|that|there|what|he|she|who|let|here|where|how)['’]s\b",
    re.IGNORECASE,
)


def contraction_count(text: str) -> int:
    return len(_CONTRACTION_RE.findall(text or ""))


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sentences(text: str) -> List[str]:
    from humanizer.text import Document

    return [s.text for s in Document.parse(text or "").sentences]


def connective_openers(text: str) -> Dict[str, int]:
    """Sentence-initial connectives.

    ``formal``  the set `transforms.strip_formal_connectives` deletes
                (Furthermore, Moreover, Therefore, ...).
    ``any``     every sentence `kriukow.opener_class` calls ``connective``,
                which adds the contrastive and enumerative openers (However,
                For example, First, ...).
    """
    from humanizer.humanize import kriukow
    from humanizer.humanize.transforms import STRIPPABLE_CONNECTIVES

    sents = _sentences(text)
    formal = 0
    any_conn = 0
    for s in sents:
        low = s.strip().lower()
        if kriukow.opener_class(s) == kriukow.OPENER_CONNECTIVE:
            any_conn += 1
        for conn in STRIPPABLE_CONNECTIVES:
            if low.startswith(conn) and (len(low) == len(conn) or not low[len(conn)].isalpha()):
                formal += 1
                break
    return {"formal": formal, "any": any_conn, "n_sentences": len(sents)}


def sentence_length_cv(text: str) -> Optional[float]:
    """`sent_len_cv` from `features.shape` (numpy only).

    Not `features.extract_features`: that also runs `structure.extract_syntax`,
    which loads spaCy and, through thinc, torch. Same number, same key.
    """
    from humanizer.features import shape
    from humanizer.text import Document

    v = _finite(shape.extract(Document.parse(text or "")).get("sent_len_cv"))
    return None if v is None else round(v, 4)


def n_words(text: str) -> int:
    return len((text or "").split())


def junk_share(text: str) -> float:
    """Share of alphabetic characters outside the Latin scripts, plus runs of
    exclamation marks. A base-model completion that runs past its paragraph
    degenerates into multilingual token noise; the fidelity gates miss it
    because `text.words` counts Latin letters only (found 2026-09-10)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    foreign = sum(1 for ch in letters if ord(ch) > 0x024F)
    bangs = text.count("!!")
    return round((foreign + bangs) / len(letters), 4)


def quality_metrics(source: str, chosen: str, facts: str = "") -> Dict[str, Any]:
    """Writing-quality numbers for the chosen text against its source."""
    from humanizer.humanize import kriukow
    from humanizer.humanize.pipeline import content_overlap, flesch_reading_ease, new_specifics

    ws, wa = n_words(source), n_words(chosen)
    fre_s, fre_a = _finite(flesch_reading_ease(source)), _finite(flesch_reading_ease(chosen))
    rep_s, rep_a = kriukow.structure_report(source), kriukow.structure_report(chosen)
    conn_s, conn_a = connective_openers(source), connective_openers(chosen)
    specifics = new_specifics(source, chosen, facts or "")
    return {
        "content_overlap": round(content_overlap(source, chosen), 4),
        "words_source": ws,
        "words_after": wa,
        "length_ratio": round(wa / ws, 4) if ws else None,
        "fre_source": None if fre_s is None else round(fre_s, 1),
        "fre_after": None if fre_a is None else round(fre_a, 1),
        "fre_delta": None if (fre_s is None or fre_a is None) else round(fre_a - fre_s, 1),
        "sent_len_cv_source": sentence_length_cv(source),
        "sent_len_cv_after": sentence_length_cv(chosen),
        "structure_flags_source": kriukow.structure_flags(rep_s),
        "structure_flags_after": kriukow.structure_flags(rep_a),
        "structure_penalty_source": rep_s.get("penalty"),
        "structure_penalty_after": rep_a.get("penalty"),
        "hedges_per_1k_source": rep_s.get("hedges_per_1k"),
        "hedges_per_1k_after": rep_a.get("hedges_per_1k"),
        "connective_openers_source": conn_s,
        "connective_openers_after": conn_a,
        "contractions_source": contraction_count(source),
        "contractions_after": contraction_count(chosen),
        "new_specifics": specifics,
        "n_new_specifics": len(specifics),
        "junk_share": junk_share(chosen),
        "lines_after": chosen.count("\n") + 1,
    }


# ------------------------------------------------------------- GPTZero side


def cache_key(text: str, multilingual: bool = False) -> str:
    """Same hash `GPTZeroClient._cache_path` uses, so we can test for a miss
    without importing the client (candidates.py prices a run before --yes)."""
    return hashlib.sha256((text + "|multilingual=%s" % multilingual).encode("utf-8")).hexdigest()


def cache_has(text: str, cache_dir: Path = CACHE_DIR) -> bool:
    return (Path(cache_dir) / (cache_key(text) + ".json")).exists()


def read_cache_payload(text: str, cache_dir: Path = CACHE_DIR) -> Optional[Dict[str, Any]]:
    p = Path(cache_dir) / (cache_key(text) + ".json")
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def parse_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """`{ai_probability, label, class_probabilities}` from a raw cache file.

    Same rule as `GPTZeroClient.parse`: ai_probability is 1 - P(human), so the
    mixed class counts against a text; label is GPTZero's own argmax verdict.
    """
    docs = payload.get("documents") or []
    doc = docs[0] if docs else {}
    probs = doc.get("class_probabilities") or {}
    p: Optional[float] = None
    if probs.get("human") is not None:
        p = 1.0 - float(probs["human"])
    elif probs.get("ai") is not None:
        p = float(probs["ai"])
    elif doc.get("completely_generated_prob") is not None:
        p = float(doc["completely_generated_prob"])
    cls = (doc.get("document_classification") or doc.get("predicted_class") or "").upper()
    label: Optional[str] = None
    if "HUMAN" in cls:
        label = "human"
    elif "MIXED" in cls:
        label = "mixed"
    elif "AI" in cls:
        label = "ai"
    return {"ai_probability": p, "label": label, "class_probabilities": probs}


def is_human(p: Optional[float], label: Optional[str]) -> Optional[bool]:
    """GPTZero's verdict when it gave one, else the 0.5 threshold; None if unscored."""
    if label in ("human", "ai", "mixed"):
        return label == "human"
    if p is None:
        return None
    return p < HUMAN_THRESHOLD


class CostLedger:
    """Counts GPTZero calls and the words that were actually charged.

    A call is charged only when the text was not already in the on-disk cache
    (`GPTZeroClient.cached`). `snapshot()`/`charge_new_files()` catch the calls
    the pipeline itself makes in own-key mode by diffing the cache directory.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.calls = 0
        self.misses = 0
        self.words_charged = 0
        self._seen: Set[str] = set()

    def snapshot(self) -> Set[str]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return set(os.listdir(self.cache_dir))

    def charge_new_files(self, before: Set[str]) -> int:
        """Charge every cache file created since `before`; returns how many."""
        added = 0
        for name in self.snapshot() - before:
            if name in self._seen:
                continue
            self._seen.add(name)
            try:
                payload = json.loads((self.cache_dir / name).read_text())
                text = ((payload.get("documents") or [{}])[0].get("inputText")) or ""
            except Exception:  # noqa: BLE001
                text = ""
            self.calls += 1
            self.misses += 1
            self.words_charged += n_words(text)
            added += 1
        return added

    def record(self, text: str, was_cached: bool) -> None:
        self.calls += 1
        if not was_cached:
            key = cache_key(text) + ".json"
            if key not in self._seen:
                self._seen.add(key)
                self.misses += 1
                self.words_charged += n_words(text)

    @property
    def usd(self) -> float:
        return round(self.words_charged * COST_PER_WORD_USD, 4)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_misses": self.misses,
            "words_charged": self.words_charged,
            "estimated_usd": self.usd,
            "rate_usd_per_word": COST_PER_WORD_USD,
        }


def score_with_retry(client: Any, text: str, ledger: Optional[CostLedger] = None,
                     max_wait_s: float = 1800.0) -> Dict[str, Any]:
    """Score one text with a `GPTZeroClient`, waiting out network outages.

    Same posture as scripts/gptzero_bench.py: retry every 60 s for up to
    `max_wait_s`, but never retry a 4xx. Texts under GPTZero's 250-character
    minimum are reported as unscored rather than raising.
    """
    words = n_words(text)
    if len(text or "") < MIN_CHARS:
        return {"ai_probability": None, "label": None, "cached": True, "words": words,
                "error": "too_short_for_gptzero"}
    was_cached = client.cached(text) is not None
    deadline = time.monotonic() + max_wait_s
    while True:
        try:
            res = client.score(text)
            break
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection errors of every shape
            msg = str(exc)
            if time.monotonic() > deadline or re.match(r"^\s*4\d\d\b", msg) or "40" in msg[:3]:
                raise
            print("  network: %s; retrying in 60 s" % type(exc).__name__, flush=True)
            time.sleep(60)
    if ledger is not None:
        ledger.record(text, was_cached)
    p = _finite(res.ai_probability)
    raw = getattr(res, "raw", None) or {}
    return {
        "ai_probability": p,
        "label": getattr(res, "label", None),
        "cached": was_cached,
        "words": words,
        "class_probabilities": raw.get("class_probabilities"),
    }


# ----------------------------------------------------------- row assembly


def _candidate_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten `as_dict()["paragraphs"][i]["candidates"]` with a `chosen` flag."""
    out: List[Dict[str, Any]] = []
    for para in result.get("paragraphs") or []:
        winner = para.get("winning_candidate")
        for c in para.get("candidates") or []:
            text = c.get("text") or ""
            proxy = c.get("proxy_probability")
            if proxy is None:
                proxy = c.get("ai_probability")
            gates = list(c.get("gates") or [])
            q = c.get("quality") or {}
            out.append({
                "paragraph_index": c.get("paragraph_index", para.get("index")),
                "index": c.get("index"),
                "round": c.get("round"),
                "style": c.get("style"),
                "persona": c.get("persona"),
                "constraint": c.get("constraint"),
                "temperature": c.get("temperature"),
                "proxy_probability": _finite(proxy),
                "gates": gates,
                "passed": bool(c.get("passed", not gates)),
                "chosen": winner is not None and c.get("index") == winner
                and c.get("paragraph_index", para.get("index")) == para.get("index"),
                "n_words": n_words(text),
                "content_overlap": q.get("content_overlap"),
                "sent_len_cv": q.get("sent_len_cv"),
                "flesch_reading_ease": q.get("flesch_reading_ease"),
                "quality_flags": list(q.get("flags") or []),
                "text": text,
            })
    return out


def assemble_row(item: Dict[str, str], result: Dict[str, Any],
                 before: Optional[Dict[str, Any]], after: Optional[Dict[str, Any]],
                 seconds: Optional[float] = None) -> Dict[str, Any]:
    """One results row from `LlmHumanizeResult.as_dict()` plus GPTZero scores.

    Works from the dict form so the self-test can feed a hand-built result.
    """
    source = item["text"]
    chosen = result.get("humanized") or source
    summary = result.get("summary") or {}
    paragraphs = result.get("paragraphs") or []
    before = before or {}
    after = after or {}
    p_before, p_after = before.get("ai_probability"), after.get("ai_probability")
    label_before, label_after = before.get("label"), after.get("label")
    candidates = _candidate_rows(result)
    row: Dict[str, Any] = {
        "name": item["name"],
        "kind": item["kind"],
        "changed": chosen.strip() != source.strip(),
        "p_before": p_before,
        "p_after": p_after,
        "label_before": label_before,
        "label_after": label_after,
        "human_before": is_human(p_before, label_before),
        "human_after": is_human(p_after, label_after),
        "gptzero_cached_before": before.get("cached"),
        "gptzero_cached_after": after.get("cached"),
        "source": source,
        "chosen": chosen,
        "facts": item.get("facts", ""),
        "seconds": None if seconds is None else round(seconds, 1),
        "pipeline_seconds": summary.get("seconds"),
        "generation_seconds": summary.get("generation_seconds"),
        # the pipeline's own yardstick view of the document (GPTZero in own-key
        # mode, the surrogate in free mode, where it is worth comparing with p_after)
        "yardstick_before": (result.get("before") or {}).get("ai_probability"),
        "yardstick_after": (result.get("after") or {}).get("ai_probability"),
        "yardstick_label_after": (result.get("after") or {}).get("label"),
        "proxy_before": paragraphs[0].get("ai_probability_before") if paragraphs else None,
        "proxy_after": paragraphs[0].get("ai_probability_after") if paragraphs else None,
        "candidate_scorer": summary.get("candidate_scorer"),
        "yardstick": (summary.get("yardstick") or {}).get("name") if isinstance(summary.get("yardstick"), dict) else summary.get("yardstick"),
        "fallback_reason": "; ".join(p.get("fallback_reason") for p in paragraphs if p.get("fallback_reason")) or None,
        "unverified_specifics": list(summary.get("unverified_specifics") or []),
        "n_candidates": len(candidates),
        "n_passed": sum(1 for c in candidates if c["passed"]),
        "gate_rejections": dict(summary.get("gate_rejections") or {}),
        "repairs_attempted": summary.get("repairs_attempted", 0),
        "repairs_accepted": summary.get("repairs_accepted", 0),
        "repairs_rescued": summary.get("repairs_rescued", 0),
        "repair_log": list(summary.get("repair_log") or []),
        "quality": quality_metrics(source, chosen, item.get("facts", "")),
        "candidates": candidates,
    }
    return row


# ---------------------------------------------------------------- summary


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.mean(vals), 3) if vals else None


def summarize(rows: Sequence[Dict[str, Any]], ledger: Optional[CostLedger] = None,
              mode: Optional[str] = None) -> Dict[str, Any]:
    ai = [r for r in rows if r["kind"] == "ai"]
    hu = [r for r in rows if r["kind"] == "human"]
    ai_scored = [r for r in ai if r["human_after"] is not None]
    hu_scored = [r for r in hu if r["human_after"] is not None]
    changed_ai = [r for r in ai if r["changed"]]
    cands = [c for r in rows for c in r["candidates"]]
    ai_cands = [c for r in ai for c in r["candidates"]]
    out: Dict[str, Any] = {
        "n_paragraphs": len(rows),
        "ai_paragraphs": len(ai),
        "ai_paragraphs_scored": len(ai_scored),
        "ai_paragraphs_human": sum(1 for r in ai_scored if r["human_after"]),
        "ai_paragraphs_changed": len(changed_ai),
        "human_paragraphs": len(hu),
        "human_paragraphs_scored": len(hu_scored),
        "humans_unharmed": sum(1 for r in hu_scored if r["human_after"]),
        "mean_p_after_ai": _mean(r["p_after"] for r in ai),
        # quality, over the AI paragraphs that were actually rewritten
        "mean_content_overlap": _mean(r["quality"]["content_overlap"] for r in changed_ai),
        "mean_length_ratio": _mean(r["quality"]["length_ratio"] for r in changed_ai),
        "mean_fre_delta": _mean(r["quality"]["fre_delta"] for r in changed_ai),
        "mean_cv_source": _mean(r["quality"]["sent_len_cv_source"] for r in changed_ai),
        "mean_cv_after": _mean(r["quality"]["sent_len_cv_after"] for r in changed_ai),
        "total_new_specifics": sum(r["quality"]["n_new_specifics"] for r in changed_ai),
        "junk_paragraphs": sum(1 for r in changed_ai if r["quality"].get("junk_share", 0) > 0.02 or r["quality"].get("lines_after", 1) > 1),
        "formal_connective_openers_source": sum(r["quality"]["connective_openers_source"]["formal"] for r in changed_ai),
        "formal_connective_openers_after": sum(r["quality"]["connective_openers_after"]["formal"] for r in changed_ai),
        "contractions_after": sum(r["quality"]["contractions_after"] for r in changed_ai),
        "structure_flags_after": _count_flags(r["quality"]["structure_flags_after"] for r in changed_ai),
        # candidates
        "n_candidates": len(cands),
        "n_candidates_passed": sum(1 for c in cands if c["passed"]),
        "gate_rejections": _gate_counts(cands),
        "mean_seconds_per_paragraph": _mean(r["seconds"] for r in rows),
        "total_seconds": round(sum(r["seconds"] or 0.0 for r in rows), 1),
    }
    if mode == "ownkey":
        # In own-key mode the candidate scorer *is* GPTZero, so every
        # candidate's proxy_probability is 1 - P(human) from GPTZero and the
        # candidate-level human rate of research/24 6.6 needs no extra calls.
        scored = [c for c in ai_cands if c["proxy_probability"] is not None]
        human = [c for c in scored if c["proxy_probability"] < HUMAN_THRESHOLD]
        both = [c for c in human if c["passed"]]
        out["candidate_level"] = {
            "source": "proxy_probability (GPTZero in own-key mode)",
            "scored": len(scored),
            "human": len(human),
            "human_share": round(len(human) / len(scored), 3) if scored else None,
            "passed_and_human": len(both),
            "passed_and_human_share": round(len(both) / len(scored), 3) if scored else None,
            "paragraphs_with_a_passed_human_candidate": sum(
                1 for r in ai if any(c["passed"] and c["proxy_probability"] is not None
                                     and c["proxy_probability"] < HUMAN_THRESHOLD for c in r["candidates"])),
        }
    else:
        out["candidate_level"] = {
            "source": "not measured: candidates were scored by the local proxy; run candidates.py --score-candidates",
        }
    if ledger is not None:
        out["gptzero_cost"] = ledger.as_dict()
    return out


def _count_flags(flag_lists: Iterable[Sequence[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for flags in flag_lists:
        for f in flags:
            counts[f] = counts.get(f, 0) + 1
    return dict(sorted(counts.items()))


def _gate_counts(cands: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in cands:
        for g in c["gates"]:
            counts[g] = counts.get(g, 0) + 1
    return dict(sorted(counts.items()))


# ------------------------------------------------------------------ tables


def fmt_p(p: Optional[float]) -> str:
    return "  -  " if p is None else "%.3f" % p


def fmt_f(v: Optional[float], spec: str = "%.2f") -> str:
    return "-" if v is None else spec % v


def text_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], right: Sequence[int] = ()) -> str:
    cols = [[str(h)] + [str(r[i]) if i < len(r) else "" for r in rows] for i, h in enumerate(headers)]
    widths = [max(len(c) for c in col) for col in cols]
    def line(cells: Sequence[Any]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            s = str(cell)
            parts.append(s.rjust(widths[i]) if i in right else s.ljust(widths[i]))
        return "  ".join(parts).rstrip()
    out = [line(headers), line(["-" * w for w in widths])]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def paragraph_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["paragraph", "kind", "before", "after", "verdict", "chg", "ov", "len",
               "FRE s>a", "CV s>a", "conn", "contr", "spec", "junk", "flags", "pass/n", "s"]
    body = []
    for r in rows:
        q = r["quality"]
        body.append([
            r["name"][:22], r["kind"], fmt_p(r["p_before"]), fmt_p(r["p_after"]),
            (r["label_after"] or ("human" if r["human_after"] else "ai" if r["human_after"] is False else "-")),
            "Y" if r["changed"] else "n",
            fmt_f(q["content_overlap"]), fmt_f(q["length_ratio"]),
            "%s>%s" % (fmt_f(q["fre_source"], "%.0f"), fmt_f(q["fre_after"], "%.0f")),
            "%s>%s" % (fmt_f(q["sent_len_cv_source"]), fmt_f(q["sent_len_cv_after"])),
            "%d>%d" % (q["connective_openers_source"]["formal"], q["connective_openers_after"]["formal"]),
            "%d>%d" % (q["contractions_source"], q["contractions_after"]),
            str(q["n_new_specifics"]),
            ("%.2f" % q.get("junk_share", 0)) + ("+" if q.get("lines_after", 1) > 1 else ""),
            ",".join(_short_flag(f) for f in q["structure_flags_after"]) or "-",
            "%d/%d" % (r["n_passed"], r["n_candidates"]),
            fmt_f(r["seconds"], "%.0f"),
        ])
    return text_table(headers, body, right=(2, 3, 6, 7, 12, 13, 16))


_FLAG_SHORT = {
    "repeated_openers": "rep",
    "uniform_beats": "beat",
    "paired_coordination_template": "pair",
    "unhedged_absolutes": "abs",
    "over_hedged": "hedg",
    "surface_statements": "surf",
}


def _short_flag(flag: str) -> str:
    return _FLAG_SHORT.get(flag, flag)


def summary_line(s: Dict[str, Any]) -> str:
    cost = s.get("gptzero_cost") or {}
    parts = [
        "AI paragraphs human %d/%d" % (s["ai_paragraphs_human"], s["ai_paragraphs"]),
        "humans unharmed %d/%d" % (s["humans_unharmed"], s["human_paragraphs"]),
        "mean overlap %s" % fmt_f(s["mean_content_overlap"]),
        "mean length ratio %s" % fmt_f(s["mean_length_ratio"]),
        "mean FRE delta %s" % fmt_f(s["mean_fre_delta"], "%+.1f"),
        "new specifics %d" % s["total_new_specifics"],
        "junk paragraphs %d" % s.get("junk_paragraphs", 0),
        "candidates passed %d/%d" % (s["n_candidates_passed"], s["n_candidates"]),
    ]
    cl = s.get("candidate_level") or {}
    if cl.get("human_share") is not None:
        parts.append("candidate-level human %d/%d (%.0f%%), passed+human %d" % (
            cl["human"], cl["scored"], 100 * cl["human_share"], cl["passed_and_human"]))
    if cost:
        parts.append("GPTZero est. $%.2f (%d calls, %d cache misses, %d words)" % (
            cost["estimated_usd"], cost["calls"], cost["cache_misses"], cost["words_charged"]))
    return "; ".join(parts)


# ------------------------------------------------------------ results io


def write_results(path: Path, meta: Dict[str, Any], rows: Sequence[Dict[str, Any]],
                  summary: Optional[Dict[str, Any]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(meta)
    payload["paragraphs"] = list(rows)
    payload["summary"] = summary
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def load_results(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
