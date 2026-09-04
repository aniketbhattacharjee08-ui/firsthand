"""Clause-level and phrasal features from a dependency parse.

Optional backend. Requires the `syntax` extra:

    pip install -e ".[syntax]"
    python -m spacy download en_core_web_sm

This module closes the gap flagged in the README validation table. The regex
heuristics in `register.py` undercounted nominalization (57.6 measured against
a 61-72 published target) because a suffix match cannot tell a nominalized noun
from an adjective or a preposition that happens to end in the same letters. A
part-of-speech tag can.

**The finding this module exists to measure.** research/15 and research/09 are
emphatic that *academic complexity is phrasal, not clausal*. There are about
twice as many dependent clauses in conversation as in academic writing, and
academic prose gets its density from noun phrases instead: attributive
adjectives, noun-noun premodification, and prepositional postmodifiers at
roughly 68 per 1,000 words. A humanizer that adds subordination to sound
scholarly is moving text toward conversation and away from the target register.

So this module reports the clausal and phrasal families separately, and adds a
`phrasal_to_clausal_ratio` so that trade can be watched directly rather than
inferred.

Published targets from research/15, per 1,000 words:
    passives                    ~18.5 (about 25% of finite verbs)
    prepositional postmodifiers ~68
    nominalizations             61.0-72.1 by sub-register
    noun-noun premodification   76.6 specialist science vs 24.3 history
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Iterable, List, Optional

from ..text import Document

MODEL_NAME = "en_core_web_sm"

# Dependency labels that introduce a dependent clause.
DEPENDENT_CLAUSE_DEPS = frozenset(
    {"ccomp", "xcomp", "advcl", "acl", "relcl", "csubj", "csubjpass", "pcomp"}
)
# Labels marking a clausal subject, used to identify finite clauses.
SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"})

# Suffixes that mark a deverbal or deadjectival noun. Applied only to tokens
# already tagged NOUN, which is the whole advantage over the regex version:
# the regex counted adjectives such as "clinical" and "critical" and so
# overshot, while a part-of-speech filter does not.
#
# "-ing" is deliberately excluded. It adds only ~4.6 per 1,000 words and drags
# in false positives like "spring", so it costs accuracy for almost no recall.
NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ancy", "ency",
    "ism", "ship", "hood", "age", "ure", "al", "ery", "sis",
)

# Only words whose ending is not a derivational suffix at all. Kept
# deliberately short: an earlier, longer stoplist excluded genuine
# nominalizations such as "activity", "quality" and "society" and pushed the
# measured rate down to 37 per 1,000 against a 61-72 target.
NOMINALIZATION_STOPLIST = frozenset(
    {
        "age", "image", "village", "message", "package", "language", "stage",
        "damage", "usage", "storage", "coverage", "percentage", "average",
        "animal", "material", "capital", "signal", "hospital", "journal",
        "metal", "total", "manual", "trial", "goal", "meal", "deal", "detail",
        "nature", "future", "figure", "picture", "culture", "structure",
        "temperature", "literature", "creature", "mixture", "feature",
        "measure", "pressure", "moisture", "furniture", "procedure", "failure",
        "city", "county", "party", "duty", "surgery", "recovery", "battery",
        "delivery", "discovery", "series", "basis", "crisis", "analysis",
        "diagnosis", "emphasis", "hypothesis",
    }
)


class SpacyUnavailable(ImportError):
    """spaCy or its English model is not installed."""


@functools.lru_cache(maxsize=2)
def load_model(name: str = MODEL_NAME):
    """Load and cache the spaCy pipeline.

    The entity recogniser and lemmatiser are disabled: nothing here needs them
    and they are a large share of parse time on long documents.
    """
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - exercised via available()
        raise SpacyUnavailable(
            "spaCy is not installed. Install the extra: pip install -e '.[syntax]'"
        ) from exc
    try:
        return spacy.load(name, disable=["ner", "lemmatizer"])
    except OSError as exc:
        raise SpacyUnavailable(
            f"spaCy model {name!r} is not installed. Run: "
            f"python -m spacy download {name}"
        ) from exc


def available(name: str = MODEL_NAME) -> bool:
    """Whether the parser backend can run right now."""
    try:
        load_model(name)
    except SpacyUnavailable:
        return False
    return True


def _is_nominalization(token) -> bool:
    if token.pos_ != "NOUN":
        return False
    lower = token.text.lower()
    if len(lower) < 5 or lower in NOMINALIZATION_STOPLIST:
        return False
    return lower.endswith(NOMINALIZATION_SUFFIXES)


def _is_finite_clause_head(token) -> bool:
    """A verb or predicate that heads a clause with its own subject."""
    if token.pos_ not in ("VERB", "AUX"):
        return False
    return any(child.dep_ in SUBJECT_DEPS for child in token.children)


def _t_unit_count(sent) -> int:
    """T-units: one main clause plus everything subordinate to it.

    Approximated as the root plus any clause-level coordinate verbs, which is
    the standard operationalisation in the L2SCA line of work.
    """
    count = 0
    for token in sent:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX", "NOUN", "ADJ"):
            count += 1
        elif token.dep_ == "conj" and token.head.dep_ == "ROOT":
            if any(child.dep_ in SUBJECT_DEPS for child in token.children):
                count += 1
    return max(count, 1)


def _is_root(token) -> bool:
    """Whether a token is its own head.

    Compare indices, never object identity. spaCy materialises a fresh Token
    object on every `.head` access, so `token.head is token` is False even at
    the root and an identity-based walk never terminates.
    """
    return token.head.i == token.i


def _dependency_depth(sent) -> int:
    depth = 0
    for token in sent:
        current, steps = token, 0
        while not _is_root(current) and steps < 100:
            current = current.head
            steps += 1
        depth = max(depth, steps)
    return depth


def extract(doc: Document, model: str = MODEL_NAME) -> Dict[str, float]:
    """Clause-level and phrasal features. Returns {} if the parser is absent."""
    try:
        nlp = load_model(model)
    except SpacyUnavailable:
        return {}
    if not doc.text.strip():
        return {}
    return extract_from_text(doc.text, nlp=nlp)


def extract_from_text(text: str, nlp=None, model: str = MODEL_NAME) -> Dict[str, float]:
    if nlp is None:
        nlp = load_model(model)

    # spaCy's default max_length guards against memory blowups on huge inputs.
    parsed = nlp(text[: nlp.max_length])
    sentences = [s for s in parsed.sents if any(not t.is_space for t in s)]
    if not sentences:
        return {}

    n_words = sum(1 for t in parsed if not t.is_punct and not t.is_space)
    if n_words == 0:
        return {}
    per_1k = 1000.0 / n_words

    finite_clauses = 0
    dependent_clauses = 0
    t_units = 0
    t_unit_words = 0
    passive_count = 0
    agentless_passive = 0
    finite_verbs = 0
    participial_tails = 0
    nominalizations = 0
    noun_compounds = 0
    attributive_adjectives = 0
    prep_postmodifiers = 0
    of_phrases = 0
    relative_clauses = 0
    adverbial_clauses = 0
    complement_clauses = 0
    coordinated_clauses = 0
    depths: List[int] = []
    pos_counts: Dict[str, int] = {}

    for sent in sentences:
        sent_t_units = _t_unit_count(sent)
        t_units += sent_t_units
        t_unit_words += sum(1 for t in sent if not t.is_punct and not t.is_space)
        depths.append(_dependency_depth(sent))

        # A sentence-final participial clause: the "-ing" tail research/04
        # reports at 5.3x the human rate with Cohen's d = 1.38.
        content = [t for t in sent if not t.is_space]
        if len(content) > 3:
            tail = content[-1] if not content[-1].is_punct else content[-2]
            head, steps = tail, 0
            while (
                not _is_root(head)
                and head.dep_ not in DEPENDENT_CLAUSE_DEPS
                and steps < 100
            ):
                head = head.head
                steps += 1
            if (
                head.dep_ in ("advcl", "acl")
                and head.tag_ == "VBG"
                and head.i > sent.start
                and any(t.text == "," and t.i < head.i for t in sent)
            ):
                participial_tails += 1

        for token in sent:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

            if _is_finite_clause_head(token):
                finite_clauses += 1
            if token.dep_ in DEPENDENT_CLAUSE_DEPS:
                dependent_clauses += 1
                if token.dep_ == "relcl":
                    relative_clauses += 1
                elif token.dep_ == "advcl":
                    adverbial_clauses += 1
                elif token.dep_ in ("ccomp", "xcomp"):
                    complement_clauses += 1
            if token.dep_ == "conj" and token.pos_ in ("VERB", "AUX"):
                coordinated_clauses += 1

            if token.dep_ == "auxpass":
                passive_count += 1
                # An agentless passive has no "by" phrase. research/04 reports
                # AI text at about half the human agentless rate, making this a
                # feature to ADD rather than remove.
                if not any(child.dep_ == "agent" for child in token.head.children):
                    agentless_passive += 1
            if token.pos_ in ("VERB", "AUX") and token.tag_ in (
                "VBD", "VBP", "VBZ", "MD",
            ):
                finite_verbs += 1

            if _is_nominalization(token):
                nominalizations += 1
            if token.dep_ == "compound" and token.pos_ in ("NOUN", "PROPN"):
                if token.head.pos_ in ("NOUN", "PROPN"):
                    noun_compounds += 1
            if token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN"):
                attributive_adjectives += 1
            if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN"):
                prep_postmodifiers += 1
                if token.text.lower() == "of":
                    of_phrases += 1

    feats: Dict[str, float] = {}

    # --- clausal complexity (the L2SCA family) ---------------------------
    feats["syn_clauses_per_1k"] = finite_clauses * per_1k
    feats["syn_dependent_clauses_per_1k"] = dependent_clauses * per_1k
    feats["syn_mean_length_t_unit"] = t_unit_words / t_units if t_units else float("nan")
    feats["syn_clauses_per_t_unit"] = (
        finite_clauses / t_units if t_units else float("nan")
    )
    feats["syn_dependent_clause_ratio"] = (
        dependent_clauses / finite_clauses if finite_clauses else float("nan")
    )
    feats["syn_relative_clause_per_1k"] = relative_clauses * per_1k
    feats["syn_adverbial_clause_per_1k"] = adverbial_clauses * per_1k
    feats["syn_complement_clause_per_1k"] = complement_clauses * per_1k
    feats["syn_coordinated_clause_per_1k"] = coordinated_clauses * per_1k

    # --- phrasal complexity (what academic prose actually uses) ----------
    feats["syn_noun_compound_per_1k"] = noun_compounds * per_1k
    feats["syn_attributive_adj_per_1k"] = attributive_adjectives * per_1k
    feats["syn_prep_postmodifier_per_1k"] = prep_postmodifiers * per_1k
    feats["syn_of_phrase_per_1k"] = of_phrases * per_1k
    feats["syn_nominalization_per_1k"] = nominalizations * per_1k

    phrasal = noun_compounds + attributive_adjectives + prep_postmodifiers
    feats["syn_phrasal_to_clausal_ratio"] = (
        phrasal / dependent_clauses if dependent_clauses else float("nan")
    )

    # --- voice ------------------------------------------------------------
    feats["syn_passive_per_1k"] = passive_count * per_1k
    feats["syn_agentless_passive_per_1k"] = agentless_passive * per_1k
    feats["syn_passive_share_of_finite"] = (
        passive_count / (finite_verbs + passive_count)
        if (finite_verbs + passive_count)
        else float("nan")
    )

    # --- other ------------------------------------------------------------
    feats["syn_participial_tail_per_1k"] = participial_tails * per_1k
    feats["syn_participial_tail_share"] = participial_tails / len(sentences)
    feats["syn_dependency_depth_mean"] = (
        sum(depths) / len(depths) if depths else float("nan")
    )
    feats["syn_dependency_depth_max"] = float(max(depths)) if depths else float("nan")

    total_pos = sum(pos_counts.values()) or 1
    for tag in ("NOUN", "VERB", "ADJ", "ADV", "ADP", "PRON", "DET", "CCONJ", "SCONJ"):
        feats[f"syn_pos_{tag.lower()}_share"] = pos_counts.get(tag, 0) / total_pos

    # Noun-to-verb ratio. research/15: news and academic prose run 3-4 nouns
    # per lexical verb, conversation runs about 1.
    verbs = pos_counts.get("VERB", 0)
    feats["syn_noun_verb_ratio"] = (
        pos_counts.get("NOUN", 0) / verbs if verbs else float("nan")
    )

    return feats


# Published human targets, used by band_report.
HUMAN_PASSIVE_PER_1K = (12.0, 26.0)
HUMAN_NOMINALIZATION_PER_1K = (55.0, 80.0)
HUMAN_PREP_POSTMODIFIER_PER_1K = (55.0, 85.0)
HUMAN_NOUN_VERB_RATIO = (2.5, 4.5)
HUMAN_DEPENDENT_CLAUSE_RATIO = (0.2, 0.6)


def band_report(feats: Dict[str, float]) -> Dict[str, str]:
    """Compare parsed features against the research/15 academic targets."""
    checks = {
        "syn_passive_per_1k": HUMAN_PASSIVE_PER_1K,
        "syn_nominalization_per_1k": HUMAN_NOMINALIZATION_PER_1K,
        "syn_prep_postmodifier_per_1k": HUMAN_PREP_POSTMODIFIER_PER_1K,
        "syn_noun_verb_ratio": HUMAN_NOUN_VERB_RATIO,
        "syn_dependent_clause_ratio": HUMAN_DEPENDENT_CLAUSE_RATIO,
    }
    out: Dict[str, str] = {}
    for key, band in checks.items():
        value = feats.get(key)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            out[key] = "unknown"
        elif value < band[0]:
            out[key] = "low"
        elif value > band[1]:
            out[key] = "high"
        else:
            out[key] = "in band"
    return out
