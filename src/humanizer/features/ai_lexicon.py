"""Weighted AI-vocabulary lexicon.

research/04 ranks AI-vocabulary density as the single highest-value feature:
the largest published effect sizes (10x to 270x excess frequency), the top cue
for expert human raters, and one of the few lexical signals that survives
paraphrasing.

Weights are the published excess-frequency ratios where a study reports one
(Kobak et al. 2024 "excess vocabulary"; Liang et al. 2024; Reinhart et al. 2025;
GPTZero's AI Vocabulary), otherwise a conservative default. A weight of 28.0
means the word appeared ~28x more often in post-LLM text than the pre-LLM
baseline predicted.

Caveat carried from research/06: GPTZero states its AI Vocabulary feature does
not itself move the probability score. Removing these words is cosmetic for the
verdict but removes the red highlights a human grader sees, and research/09
scores the edit as zero-to-positive for the essay grade. So it stays in the
"do aggressively" tier for product reasons, not detector reasons.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

DEFAULT_WEIGHT = 3.0

# Single words. Ratios from Kobak et al. where published.
WORD_WEIGHTS: Dict[str, float] = {
    # Kobak et al. 2024, highest published excess ratios
    "delves": 28.0,
    "delve": 25.0,
    "delving": 20.0,
    "underscores": 13.8,
    "underscore": 12.0,
    "underscoring": 11.0,
    "showcasing": 10.7,
    "showcase": 9.0,
    "showcases": 9.0,
    "pivotal": 8.5,
    "intricate": 8.0,
    "intricacies": 8.0,
    "realm": 7.5,
    "realms": 7.0,
    "garnered": 7.0,
    "burgeoning": 6.5,
    "noteworthy": 6.0,
    "meticulous": 6.0,
    "meticulously": 6.0,
    "commendable": 6.0,
    "intricately": 5.5,
    "encompassing": 5.5,
    "multifaceted": 5.5,
    "nuanced": 5.0,
    "leverage": 5.0,
    "leveraging": 5.0,
    "harness": 4.5,
    "harnessing": 4.5,
    "unveiling": 4.5,
    "unravel": 4.5,
    "elucidate": 4.5,
    "endeavors": 4.0,
    "paramount": 4.0,
    "profound": 4.0,
    "seamless": 4.0,
    "seamlessly": 4.0,
    "robust": 3.5,
    "crucial": 3.5,
    "vital": 3.0,
    "significant": 2.0,
    "comprehensive": 3.0,
    "holistic": 4.0,
    "innovative": 3.5,
    "transformative": 4.5,
    "groundbreaking": 4.5,
    "cutting-edge": 4.0,
    "landscape": 4.0,
    "tapestry": 12.0,
    "testament": 8.0,
    "camaraderie": 12.0,
    "beacon": 6.0,
    "myriad": 5.5,
    "plethora": 5.5,
    "bolster": 4.5,
    "foster": 3.0,
    "fostering": 3.5,
    "facilitate": 3.0,
    "utilize": 3.5,
    "utilizing": 3.5,
    "utilization": 3.5,
    "ethos": 4.0,
    "interplay": 4.5,
    "synergy": 4.5,
    "trajectory": 3.5,
    "nexus": 4.5,
    "paradigm": 3.5,
    "cornerstone": 4.5,
    "hallmark": 4.0,
    "quintessential": 5.0,
    "indelible": 5.5,
    "resonate": 4.0,
    "resonates": 4.0,
    "captivating": 5.0,
    "compelling": 3.0,
    "invaluable": 4.0,
    "unwavering": 5.0,
    "steadfast": 4.5,
    "diligently": 4.5,
    "adept": 4.5,
    "adeptly": 5.0,
    "poised": 4.0,
    "underpinning": 4.0,
    "underpinned": 4.0,
    "spearheaded": 4.5,
    "align": 2.5,
    "aligns": 3.0,
    "aligning": 3.0,
    "streamline": 4.0,
    "streamlined": 4.0,
    "empower": 4.0,
    "empowering": 4.0,
    "navigate": 3.5,
    "navigating": 4.0,
    "amplify": 4.0,
    "illuminate": 4.0,
    "illuminating": 4.0,
    "vibrant": 3.5,
    "dynamic": 2.5,
    "evolving": 3.0,
    "ever-evolving": 6.0,
    "rapidly-evolving": 6.0,
    "pertinent": 3.5,
    "salient": 3.5,
    "efficacy": 3.0,
    "imperative": 4.0,
    "integral": 3.0,
    "notably": 2.5,
    "moreover": 3.0,
    "furthermore": 3.0,
    "additionally": 3.0,
    "consequently": 2.5,
    "nevertheless": 2.0,
}

# Multi-word phrases. GPTZero reports some at up to 269x enrichment.
PHRASE_WEIGHTS: Dict[str, float] = {
    "it is important to note": 15.0,
    "it's important to note": 15.0,
    "it is worth noting": 12.0,
    "it's worth noting": 12.0,
    "it is crucial to": 10.0,
    "plays a crucial role": 12.0,
    "plays a vital role": 11.0,
    "plays a significant role": 9.0,
    "plays a pivotal role": 14.0,
    "in the realm of": 14.0,
    "in the ever-evolving": 16.0,
    "in today's fast-paced": 14.0,
    "in today's digital age": 14.0,
    "navigating the complexities": 16.0,
    "a testament to": 10.0,
    "rich tapestry": 18.0,
    "delve into": 20.0,
    "delves into": 22.0,
    "shed light on": 8.0,
    "sheds light on": 8.0,
    "at the forefront": 8.0,
    "pave the way": 8.0,
    "paves the way": 8.0,
    "paving the way": 8.0,
    "the world of": 4.0,
    "when it comes to": 5.0,
    "on the other hand": 3.0,
    "in conclusion": 8.0,
    "in summary": 6.0,
    "to sum up": 6.0,
    "overall, this": 5.0,
    "as we have seen": 5.0,
    "one of the most": 3.0,
    "a wide range of": 4.0,
    "a myriad of": 10.0,
    "a plethora of": 10.0,
    "the intersection of": 6.0,
    "underscores the importance": 16.0,
    "highlights the importance": 10.0,
    "emphasizes the importance": 9.0,
    "it is essential to": 8.0,
    "essential to understand": 7.0,
    "gain a deeper understanding": 10.0,
    "a deeper understanding of": 8.0,
    "comprehensive overview": 8.0,
    "valuable insights": 9.0,
    "meaningful insights": 8.0,
    "significant implications": 6.0,
    "far-reaching implications": 8.0,
    "the key takeaway": 7.0,
    "not only": 2.5,
    "but also": 2.5,
    "foster a sense of": 9.0,
    "a crucial role in": 10.0,
    "continues to evolve": 7.0,
    "stands as a": 7.0,
    "serves as a": 5.0,
    "is a powerful tool": 8.0,
    "unlock the potential": 11.0,
    "harness the power": 12.0,
    "embark on a journey": 12.0,
    "the importance of understanding": 8.0,
    "by understanding the": 6.0,
    "ultimately, the": 5.0,
}

# Formal sentence-initial connectives (research/04 feature 9, research/09 rates
# removing these as the single best joint move: high detector benefit, and it
# *raises* the essay grade because IELTS Band 9 coherence "attracts no
# attention").
FORMAL_CONNECTIVES = {
    "additionally", "moreover", "furthermore", "consequently", "therefore",
    "thus", "hence", "nevertheless", "nonetheless", "accordingly",
    "subsequently", "similarly", "likewise", "conversely", "notably",
    "importantly", "specifically", "ultimately", "overall", "indeed",
    "in addition", "in conclusion", "in summary", "in contrast",
    "on the other hand", "as a result", "for instance", "for example",
}

INFORMAL_CONNECTIVES = {
    "but", "so", "and", "also", "still", "yet", "then", "besides",
    "anyway", "plus",
}

_WORD_BOUNDARY_CACHE: Dict[str, re.Pattern] = {}


def _phrase_pattern(phrase: str) -> re.Pattern:
    if phrase not in _WORD_BOUNDARY_CACHE:
        _WORD_BOUNDARY_CACHE[phrase] = re.compile(
            r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b",
            re.IGNORECASE,
        )
    return _WORD_BOUNDARY_CACHE[phrase]


def find_hits(text: str, lower_words: List[str]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Return (word_hits, phrase_hits) as (surface, weight) pairs."""
    word_hits = [(w, WORD_WEIGHTS[w]) for w in lower_words if w in WORD_WEIGHTS]
    phrase_hits: List[Tuple[str, float]] = []
    for phrase, weight in PHRASE_WEIGHTS.items():
        count = len(_phrase_pattern(phrase).findall(text))
        phrase_hits.extend([(phrase, weight)] * count)
    return word_hits, phrase_hits
