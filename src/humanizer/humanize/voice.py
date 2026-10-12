"""Your voice: a user's own writing as the exemplars and as a ranking signal.

Two tiers, both prompt- and ranker-level. No per-user training happens here.

**Tier 1, the exemplars.** The base-model (`freeform`) prompt is a few-shot
pattern of (draft, notes, rewrite) triples (`exemplars.FREEFORM_SHOTS`). With
a voice attached, the user's own samples take those slots: the rewrite is a
paragraph they really wrote, the draft is a generic instruct-model version of
it made once at upload time (`flatten_prompt`), and the notes are the
particulars the paragraph contains (`notes_from`). The base model continues
the pattern it is shown, so it continues *this person's* pattern.

**Tier 2, the fingerprint.** `build_profile` reduces the samples to the
features that research/16 found actually individuate authors: function-word
and punctuation rates, sentence-length spread, openers, contractions, and a
character-trigram profile. Mean sentence length is deliberately a minor term
(Mosteller and Wallace: Hamilton and Madison differ by 0.04 words). The
pipeline ranks candidates that already pass the detector gate by
`distance(profile, candidate)`; the gate stays hard, so a voice can only
choose among human-reading rewrites, never promote an AI-reading one.

**What is never done.** Model output is never added to the sample pool. A
liked rewrite goes into a separate pile (`api.voice`), is used as an
exemplar only when the user has fewer than `MIN_REAL_SHOTS` real samples,
and even then at most one of the three slots: the loop would otherwise copy
the model's habits back into the prompt, which is the one thing detectors
learn to spot.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..features import extract_features
from ..text import words
from .engine import invariants
from . import llm as llm_mod

__all__ = [
    "VoiceProfile",
    "VoiceSpec",
    "build_profile",
    "profile_distance",
    "trigram_profile",
    "trigram_distance",
    "notes_from",
    "flatten_prompt",
    "clean_flattened",
    "retained_share",
    "MAX_TWIN_RETAINED",
    "select_shots",
    "describe_profile",
    "MIN_SAMPLE_WORDS",
    "MAX_SAMPLE_WORDS",
    "PART_WORDS",
    "SHOT_WORD_BUDGET",
    "MAX_SAMPLES",
    "split_sample",
    "neutralise_markers",
    "MIN_REAL_SHOTS",
    "STRENGTHS",
]

#: Sample limits. 80 words is two short paragraphs, the least that gives
#: the feature extractor anything to count; 2,500 keeps one upload from
#: being a whole thesis that dominates the profile.
MIN_SAMPLE_WORDS = 80
MAX_SAMPLE_WORDS = 2500
#: An upload over this many words is split at paragraph boundaries into
#: parts of about this size, each its own sample: exemplars are paragraph
#: sized, and the per-paragraph picker matches them to a paragraph.
PART_WORDS = 450
MAX_SAMPLES = 12
#: The exemplar block's budget in words, samples and their twins together:
#: about 2,000 tokens, which with the header, the draft and the completion
#: stays inside the served model's 6,144-token window (deploy/cloud).
SHOT_WORD_BUDGET = 1500
#: Below this many ready real samples, one liked rewrite may fill a slot.
MIN_REAL_SHOTS = 3
#: Exemplar slots in the few-shot block, the same count as the stock set.
N_SHOTS = 3
#: How much the fingerprint counts, in units of AI probability, among the
#: candidates that already pass the gate. Passers sit at p <= 0.15, so 0.5
#: lets the voice decide among them without ever outranking the gate.
STRENGTHS: Dict[str, float] = {"light": 0.25, "balanced": 0.5, "strong": 1.0}
DEFAULT_STRENGTH = "balanced"

#: The features the fingerprint is built from, with a weight. Function
#: words, punctuation and openers carry the author signal (research/16 §2);
#: sentence-length *spread* matters, its mean barely does.
PROFILE_FEATURES: Tuple[Tuple[str, float], ...] = (
    ("sent_len_cv", 1.0),
    ("sent_len_sd", 0.6),
    ("sent_len_mean", 0.3),
    ("sent_short_share", 0.8),
    ("sent_long_share", 0.8),
    ("comma_per_1k", 1.0),
    ("semicolon_per_1k", 0.8),
    ("colon_per_1k", 0.6),
    ("paren_per_1k", 0.6),
    ("question_mark_per_1k", 0.5),
    ("contraction_per_1k", 1.0),
    ("hedge_per_1k", 0.8),
    ("booster_per_1k", 0.6),
    ("formal_connective_per_1k", 0.8),
    ("informal_connective_per_1k", 0.8),
    ("sent_initial_coordinator_share", 0.8),
    ("struct_opener_pronoun_share", 0.8),
    ("struct_opener_det_share", 0.6),
    ("struct_opener_adverbial_share", 0.6),
    ("struct_sentence_type_simple_share", 0.6),
    ("struct_sentence_type_complex_share", 0.6),
    ("self_mention_singular_per_10k", 0.8),
    ("passive_per_1k", 0.6),
    ("nominalization_per_1k", 0.6),
    ("long_word_share", 0.6),
    ("word_len_mean", 0.4),
    ("mattr_50", 0.5),
    ("punct_entropy", 0.6),
)

#: A floor on each feature's spread so one sample does not make a feature
#: infinitely sharp. Expressed as a fraction of the feature's mean, with an
#: absolute floor for rates that are often zero.
_SD_FLOOR_FRACTION = 0.25
_SD_FLOOR_ABS = 0.5


@dataclass
class VoiceProfile:
    """The fingerprint: per-feature mean and spread, plus a trigram profile."""

    n_samples: int = 0
    n_words: int = 0
    means: Dict[str, float] = field(default_factory=dict)
    sds: Dict[str, float] = field(default_factory=dict)
    #: Character trigram relative frequencies over the samples, top-K.
    trigrams: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_words": self.n_words,
            "means": dict(self.means),
            "sds": dict(self.sds),
            "trigrams": dict(self.trigrams),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "VoiceProfile":
        raw = raw or {}
        return cls(
            n_samples=int(raw.get("n_samples") or 0),
            n_words=int(raw.get("n_words") or 0),
            means={str(k): float(v) for k, v in (raw.get("means") or {}).items()},
            sds={str(k): float(v) for k, v in (raw.get("sds") or {}).items()},
            trigrams={str(k): float(v) for k, v in (raw.get("trigrams") or {}).items()},
        )

    @property
    def empty(self) -> bool:
        return self.n_samples == 0 or not self.means


@dataclass
class VoiceSpec:
    """What the pipeline receives for one run: the shots, the profile, the weight.

    `shots` are (topic, draft, notes, rewrite) tuples in the shape
    `llm.build_prompt` takes. Empty means "keep the stock exemplars and rank
    with the profile only". `sample_ids` is reported back so the feedback
    can reweight the samples that were used.
    """

    profile: Optional[VoiceProfile] = None
    shots: Tuple[Tuple[str, str, str, str], ...] = ()
    sample_ids: Tuple[str, ...] = ()
    strength: str = DEFAULT_STRENGTH
    run_id: str = ""
    #: Optional per-paragraph picker: `shots_for(paragraph) -> shots`. When
    #: set, the pipeline calls it for each paragraph instead of `shots`.
    shots_for: Optional[Callable[[str], Sequence[Tuple[str, str, str, str]]]] = None
    #: Why the voice could not be used, when it could not (`login_required`,
    #: `no_samples`, `disabled`). Reported in `summary["voice"]`.
    reason: str = ""
    #: Filled by `shots_for` as paragraphs are prompted: the ids actually
    #: used, which is what the feedback reweights and `uses` counts.
    used_ids: List[str] = field(default_factory=list)
    #: Called once at the end of a run with `used_ids`, for the store.
    on_used: Optional[Callable[[Sequence[str]], None]] = None

    @property
    def weight(self) -> float:
        return STRENGTHS.get(self.strength, STRENGTHS[DEFAULT_STRENGTH])

    @property
    def usable(self) -> bool:
        if self.reason:
            return False
        return bool(self.shots) or self.shots_for is not None or (
            self.profile is not None and not self.profile.empty
        )


# ------------------------------------------------------------------ profile


_TRIGRAM_TOP_K = 600
_TRIGRAM_RE = re.compile(r"\s+")


def trigram_profile(texts: Iterable[str], top_k: int = _TRIGRAM_TOP_K) -> Dict[str, float]:
    """Relative frequencies of the `top_k` character trigrams, lower-cased,
    whitespace collapsed to one space. Character n-grams are the strongest
    cheap authorship feature (research/16 §2, Grieve 2005)."""
    counts: Counter = Counter()
    for text in texts:
        t = _TRIGRAM_RE.sub(" ", (text or "").lower()).strip()
        if len(t) < 3:
            continue
        counts.update(t[i : i + 3] for i in range(len(t) - 2))
    total = sum(counts.values())
    if not total:
        return {}
    return {g: n / total for g, n in counts.most_common(top_k)}


def trigram_distance(profile: Dict[str, float], text: str) -> float:
    """1 minus the cosine similarity between the profile and `text`'s
    trigram distribution, in [0, 1]. 0 for an empty profile."""
    if not profile:
        return 0.0
    mine = trigram_profile([text], top_k=10 ** 6)
    if not mine:
        return 1.0
    dot = sum(v * mine.get(g, 0.0) for g, v in profile.items())
    na = math.sqrt(sum(v * v for v in profile.values()))
    nb = math.sqrt(sum(v * v for v in mine.values()))
    if na <= 0 or nb <= 0:
        return 1.0
    return float(min(1.0, max(0.0, 1.0 - dot / (na * nb))))


def _finite(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def build_profile(samples: Sequence[str]) -> VoiceProfile:
    """The fingerprint over `samples`. Empty input gives an empty profile."""
    texts = [s for s in samples if s and s.strip()]
    if not texts:
        return VoiceProfile()
    rows: List[Dict[str, float]] = []
    n_words = 0
    for text in texts:
        feats = extract_features(text)
        rows.append(feats)
        n_words += len(words(text))
    means: Dict[str, float] = {}
    sds: Dict[str, float] = {}
    for key, _w in PROFILE_FEATURES:
        values = [v for v in (_finite(r.get(key)) for r in rows) if v is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            sd = math.sqrt(var)
        else:
            sd = 0.0
        floor = max(_SD_FLOOR_ABS * _scale_for(key), abs(mean) * _SD_FLOOR_FRACTION)
        means[key] = mean
        sds[key] = max(sd, floor)
    return VoiceProfile(
        n_samples=len(texts),
        n_words=n_words,
        means=means,
        sds=sds,
        trigrams=trigram_profile(texts),
    )


def _scale_for(key: str) -> float:
    """Absolute spread floor per feature family: shares live in [0, 1],
    rates per thousand in tens, lengths in words."""
    if key.endswith("_share") or key in ("mattr_50", "sent_len_cv", "punct_entropy"):
        return 0.1
    if key.endswith("_per_10k"):
        return 4.0
    if key.endswith("_per_1k"):
        return 2.0
    return 1.0


def profile_distance(profile: Optional[VoiceProfile], text: str) -> float:
    """How far `text` sits from the profile, in [0, 1]; lower is closer.

    Two halves, averaged: the weighted mean of clipped z-scores over
    `PROFILE_FEATURES` (a z of 2 counts as the full distance), and the
    trigram cosine distance. Short texts are noisy on both, which is why
    this is only ever applied among candidates that already pass the
    detector gate.
    """
    if profile is None or profile.empty or not text or not text.strip():
        return 0.0
    feats = extract_features(text)
    num = 0.0
    den = 0.0
    for key, weight in PROFILE_FEATURES:
        mean = profile.means.get(key)
        sd = profile.sds.get(key)
        value = _finite(feats.get(key))
        if mean is None or sd is None or value is None or sd <= 0:
            continue
        z = abs(value - mean) / sd
        num += weight * min(1.0, z / 2.0)
        den += weight
    feature_term = num / den if den > 0 else 0.0
    trigram_term = trigram_distance(profile.trigrams, text)
    return float(min(1.0, max(0.0, 0.5 * feature_term + 0.5 * trigram_term)))


# --------------------------------------------------------- readable traits


def describe_profile(profile: Optional[VoiceProfile]) -> List[Dict[str, str]]:
    """Plain-words rows for the page: what the fingerprint says about the author."""
    if profile is None or profile.empty:
        return []
    m = profile.means
    rows: List[Dict[str, str]] = []

    mean_len = m.get("sent_len_mean")
    cv = m.get("sent_len_cv")
    if mean_len is not None:
        spread = "a wide spread" if (cv or 0) >= 0.5 else ("some variety" if (cv or 0) >= 0.3 else "little variation")
        rows.append(
            {
                "key": "sentences",
                "label": "Sentence length",
                "value": "%d words on average, with %s" % (round(mean_len), spread),
                "note": "Short and long sentences side by side read as a person thinking."
                if (cv or 0) >= 0.5
                else "An even rhythm. The rewrite keeps to it rather than forcing variety.",
            }
        )

    contractions = m.get("contraction_per_1k")
    if contractions is not None:
        if contractions >= 8:
            value, note = "Often", "You write don't and it's rather than do not and it is."
        elif contractions >= 2:
            value, note = "Sometimes", "A contraction now and then, mostly in asides."
        else:
            value, note = "Rarely", "You spell words out. The rewrite will too."
        rows.append({"key": "contractions", "label": "Contractions", "value": value, "note": note})

    formal = m.get("formal_connective_per_1k") or 0.0
    informal = m.get("informal_connective_per_1k") or 0.0
    coord = m.get("sent_initial_coordinator_share") or 0.0
    if formal or informal or coord:
        if coord >= 0.12 or informal > formal:
            value = "Plain: and, but, so"
            note = "Sentences often start with a short conjunction."
        elif formal >= 6:
            value = "Formal: however, therefore, moreover"
            note = "Academic connectives carry the argument."
        else:
            value = "Mixed"
            note = "Neither formal connectives nor short conjunctions dominate."
        rows.append({"key": "connectives", "label": "How sentences join", "value": value, "note": note})

    punct_bits = []
    if (m.get("semicolon_per_1k") or 0) >= 1.5:
        punct_bits.append("semicolons")
    if (m.get("colon_per_1k") or 0) >= 1.5:
        punct_bits.append("colons")
    if (m.get("paren_per_1k") or 0) >= 1.5:
        punct_bits.append("parentheses")
    if (m.get("question_mark_per_1k") or 0) >= 1.0:
        punct_bits.append("questions")
    commas = m.get("comma_per_1k")
    if commas is not None:
        if commas >= 60:
            comma_word = "heavy on commas"
        elif commas >= 35:
            comma_word = "a normal run of commas"
        else:
            comma_word = "light on commas"
        value = comma_word + (", with " + _join(punct_bits) if punct_bits else "")
        rows.append(
            {
                "key": "punctuation",
                "label": "Punctuation",
                "value": value[0].upper() + value[1:],
                "note": "The marks you reach for are among the strongest signs of who wrote a text.",
            }
        )

    hedges = m.get("hedge_per_1k") or 0.0
    boosters = m.get("booster_per_1k") or 0.0
    if hedges or boosters:
        if hedges >= boosters * 1.5 and hedges >= 4:
            value, note = "Hedged", "Perhaps, seems, might: you leave room for doubt."
        elif boosters >= hedges * 1.5 and boosters >= 4:
            value, note = "Assertive", "Clearly, certainly, always: you commit."
        else:
            value, note = "Even-handed", "Hedges and boosters in roughly equal measure."
        rows.append({"key": "stance", "label": "Stance", "value": value, "note": note})

    self_mention = m.get("self_mention_singular_per_10k") or 0.0
    rows.append(
        {
            "key": "person",
            "label": "First person",
            "value": "Present" if self_mention >= 20 else ("Occasional" if self_mention >= 5 else "Absent"),
            "note": "Whether I and my appear, and how often.",
        }
    )

    long_share = m.get("long_word_share")
    if long_share is not None:
        if long_share >= 0.22:
            value, note = "Long words", "Many words over six letters: an academic register."
        elif long_share >= 0.14:
            value, note = "Mixed", "A normal balance of short and long words."
        else:
            value, note = "Short words", "Mostly short words: plain, direct prose."
        rows.append({"key": "vocabulary", "label": "Vocabulary", "value": value, "note": note})

    return rows


def _join(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


# ------------------------------------------------------------------- notes


_NOTE_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def notes_from(text: str, limit: int = 5) -> str:
    """The particulars a sample contains, as the "- " bullets the few-shot
    pattern expects. Each sentence that carries a number, a date, a quotation,
    a citation or a capitalised run becomes one bullet, trimmed. With nothing
    particular in the text the single bullet says so, which is what the stock
    prompt does without facts."""
    bullets: List[str] = []
    for sentence in _NOTE_SENTENCE_RE.split((text or "").strip()):
        s = sentence.strip()
        if not s or len(s) < 12:
            continue
        inv = invariants(s)
        # "I" is a capitalised run to the entity finder; it is not a particular.
        entities = {e for e in inv["entities"] if len(e) > 1}
        has_particular = any(
            inv[key] for key in ("numbers", "quotations", "citations", "dates")
        ) or bool(entities)
        if not has_particular:
            continue
        if len(s) > 220:
            s = s[:217].rsplit(" ", 1)[0] + "..."
        bullets.append("- " + s)
        if len(bullets) >= limit:
            break
    if not bullets:
        return "- No particulars to add; keep the claims as they are."
    return "\n".join(bullets)


# ------------------------------------------------------------- the draft


#: The twin is made by the *base* checkpoint, the one model every
#: deployment serves (deploy/cloud/modal_vllm.py serves the base with its
#: adapters and nothing else), by continuing the stock pattern backwards:
#: three (HUMAN, DRAFT) pairs from `exemplars.FREEFORM_SHOTS`, then the
#: sample as HUMAN and an open DRAFT slot. A base model continues the
#: pattern it is shown; shown human paragraphs each followed by their
#: generic assistant version, it writes the generic version of the sample.
FLATTEN_HEADER = (
    "Each paragraph below is given twice: first as its author wrote it "
    "(HUMAN), then as a writing assistant would produce the same paragraph "
    "(DRAFT): smoother, more formal, more general, the same claims and the "
    "same particulars in the same order, nothing added.\n\n"
)
FLATTEN_STOPS = llm_mod.FREEFORM_STOPS


def flatten_prompt(sample: str) -> llm_mod.Prompt:
    """The base-model prompt that turns a real paragraph into its generic twin.

    This is the one place the product asks a model to sound *more* like an
    assistant: the twin is the draft half of the exemplar pair, and the
    user's real paragraph is the rewrite half.
    """
    from .exemplars import FREEFORM_SHOTS

    block = "".join(
        "HUMAN:\n%s\n\nDRAFT:\n%s\n\n" % (rewrite, draft) for _topic, draft, _notes, rewrite in FREEFORM_SHOTS
    )
    return llm_mod.Prompt(
        kind="text",
        text=FLATTEN_HEADER + block + "HUMAN:\n" + sample.strip() + "\n\nDRAFT:\n",
    )


#: A twin that keeps more than this share of the sample's word *bigrams* is
#: an echo, not a generic version: the pair would teach the model to copy
#: the draft. Bigrams, not words: a paraphrase of a technical paragraph
#: rightly keeps most of its terms (measured 2026-10-11 on PMC text, 0.75
#: to 0.90 of the words survive a real paraphrase) but breaks most of its
#: phrasing (0.47 to 0.66 of bigrams). The adapter checkpoint's echoes sit
#: above 0.9 on both.
MAX_TWIN_RETAINED = 0.75


def retained_share(sample: str, twin: str) -> float:
    """The share of the sample's adjacent word pairs that survive in the twin."""
    mine = [w.lower() for w in words(sample)]
    theirs = [w.lower() for w in words(twin or "")]
    pairs = list(zip(mine, mine[1:]))
    if not pairs:
        return 1.0 if mine and mine == theirs else 0.0
    have = set(zip(theirs, theirs[1:]))
    return sum(1 for pr in pairs if pr in have) / len(pairs)


def clean_flattened(raw: str, sample: str) -> Optional[str]:
    """The model's twin, or None when it is unusable (empty, a refusal, the
    sample echoed back, or wildly off in length)."""
    text = llm_mod.clean_completion(raw or "", FLATTEN_STOPS).strip()
    if not text:
        return None
    if text.strip() == sample.strip():
        return None
    n_in = len(words(sample))
    n_out = len(words(text))
    if n_in and (n_out < 0.4 * n_in or n_out > 2.0 * n_in):
        return None
    low = text.lower()
    if low.startswith(("i'm sorry", "i am sorry", "i cannot", "i can't", "as an ai")):
        return None
    return text


# ------------------------------------------------------------------ shots


def select_shots(
    samples: Sequence[Dict[str, Any]],
    paragraph: str,
    liked: Sequence[Dict[str, Any]] = (),
    k: int = N_SHOTS,
    budget: int = SHOT_WORD_BUDGET,
) -> Tuple[Tuple[Tuple[str, str, str, str], ...], Tuple[str, ...]]:
    """The exemplar triples for one paragraph, and the ids they came from.

    `samples` are ready real samples: dicts with `id`, `text`, `draft`,
    `notes`, `title`, `weight`. `liked` are liked rewrites: dicts with `id`,
    `original`, `text`. Scored by trigram similarity to the paragraph, by how
    close in length the sample is, and by the sample's weight (what thumbs
    up and down have taught), so an essay paragraph gets essay samples. One
    liked rewrite fills a slot only when fewer than `MIN_REAL_SHOTS` real
    samples are ready, and only ever one.
    """
    ready = [s for s in samples if s.get("draft") and s.get("text")]
    if not ready and not liked:
        return (), ()
    target = trigram_profile([paragraph], top_k=10 ** 6)
    n_target = max(1, len(words(paragraph)))

    def score(sample: Dict[str, Any]) -> float:
        text = sample.get("text") or ""
        sim = 1.0 - trigram_distance(target, text) if target else 0.5
        n = max(1, len(words(text)))
        length_term = 1.0 - min(1.0, abs(math.log(n / n_target)) / math.log(6))
        weight = float(sample.get("weight") or 1.0)
        return 0.6 * sim + 0.2 * length_term + 0.2 * min(2.0, max(0.0, weight)) / 2.0

    ranked = sorted(ready, key=score, reverse=True)
    chosen: List[Tuple[str, str, str, str]] = []
    ids: List[str] = []
    real_slots = k if len(ranked) >= MIN_REAL_SHOTS or not liked else max(0, k - 1)
    # A word budget over sample and twin together, so three long samples
    # cannot push the prompt past the served model's context window. The
    # best-ranked sample that fits is taken; one that does not is skipped.
    spent = 0
    for s in ranked:
        if len(chosen) >= real_slots:
            break
        cost = len(words(s["text"])) + len(words(s["draft"]))
        if chosen and spent + cost > budget:
            continue
        if not chosen and cost > budget:
            continue
        chosen.append((s.get("title") or "the author's own writing", s["draft"], s.get("notes") or notes_from(s["text"]), s["text"]))
        ids.append(str(s["id"]))
        spent += cost
    if len(chosen) < k and liked and len(ranked) < MIN_REAL_SHOTS:
        best_liked = max(liked, key=lambda l: 1.0 - trigram_distance(target, l.get("text") or "") if target else 0.0)
        liked_cost = len(words(best_liked.get("original") or "")) + len(words(best_liked.get("text") or ""))
        if best_liked.get("original") and best_liked.get("text") and spent + liked_cost <= budget:
            chosen.append(
                (
                    "a rewrite the author approved",
                    best_liked["original"],
                    notes_from(best_liked["original"]),
                    best_liked["text"],
                )
            )
            ids.append(str(best_liked["id"]))
    # The pattern reads best with the closest match last, next to the draft.
    chosen.reverse()
    ids.reverse()
    return tuple(chosen), tuple(ids)


# ---------------------------------------------------------------- uploads


_MARKER_RE = re.compile(r"(?m)^[ \t]*(DRAFT|NOTES|HUMAN|Editing log)[ \t]*:")


def neutralise_markers(text: str) -> str:
    """Defuse the few-shot pattern's own labels inside a user's text.

    A line that is exactly `HUMAN:` or `DRAFT:` would read to the base
    model as the next slot of the pattern. The colon is dropped so the
    words stay and the label does not. Only this user's own run could be
    affected, but a sample should not be able to break the pattern at all.
    """
    return _MARKER_RE.sub(lambda m: m.group(1) + " ", text or "")


def split_sample(text: str, part_words: int = PART_WORDS, min_words: int = MIN_SAMPLE_WORDS) -> List[str]:
    """Split an upload into parts of about `part_words` at paragraph (then
    sentence) boundaries. A text under the limit is returned whole; a
    trailing part shorter than `min_words` is merged into the previous one."""
    text = (text or "").strip()
    if len(words(text)) <= part_words:
        return [text] if text else []
    units: List[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(words(para)) <= part_words:
            units.append(para)
            continue
        # One very long paragraph: cut it at sentence ends instead.
        current: List[str] = []
        n = 0
        for sent in _NOTE_SENTENCE_RE.split(para):
            w = len(words(sent))
            if current and n + w > part_words:
                units.append(" ".join(current))
                current, n = [], 0
            current.append(sent)
            n += w
        if current:
            units.append(" ".join(current))
    parts: List[str] = []
    current = []
    n = 0
    for unit in units:
        w = len(words(unit))
        if current and n + w > part_words:
            parts.append("\n\n".join(current))
            current, n = [], 0
        current.append(unit)
        n += w
    if current:
        parts.append("\n\n".join(current))
    if len(parts) > 1 and len(words(parts[-1])) < min_words:
        last = parts.pop()
        parts[-1] = parts[-1] + "\n\n" + last
    return parts
