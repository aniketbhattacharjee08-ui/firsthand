#!/usr/bin/env python3
"""Fetch pre-LLM human reference corpora.

Two sources that need no credentials and no AWS client, chosen from the
acquisition table in research/14:

  wikipedia  Featured Articles at a pre-November-2022 revision. research/10
             used the 2021-06-01 revision as its "clean, edited, impersonal
             expository prose" reference.
  pmc        Open Access research articles via the Europe PMC REST API,
             restricted to 2016-2021 so nothing post-dates ChatGPT.

Both are cut off before generative models entered general use, which is the
whole point: a reference distribution contaminated with AI text would teach the
system to imitate the thing it is meant to avoid.

Usage:
    python scripts/fetch_reference.py wikipedia --count 120 --out data/raw/wikipedia
    python scripts/fetch_reference.py pmc --count 200 --out data/raw/pmc

Then:
    humanizer build-reference expository 'data/raw/wikipedia/*.txt' \
        --out data/reference/expository.json
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

USER_AGENT = "humanizer-research/0.1 (reference corpus build)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_CUTOFF = "2021-06-01T00:00:00Z"
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


class FetchError(RuntimeError):
    """A fetch failed in a way the caller should report, not retry blindly."""


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def get_json(
    session: requests.Session,
    url: str,
    params: dict,
    timeout: float = 45.0,
    retries: int = 4,
) -> dict:
    """GET returning JSON, with backoff on rate limiting.

    Public APIs return HTTP 429 with a plain-text body, which a bare .json()
    call turns into a confusing JSONDecodeError. Shared or cloud IPs hit this
    routinely, so it is handled explicitly rather than treated as a parse bug.
    """
    delay = 2.0
    last = ""
    for attempt in range(retries):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code == 429:
            last = "rate limited (HTTP 429)"
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if (retry_after or "").isdigit() else delay
            print(
                f"  rate limited, waiting {wait:.0f}s "
                f"(attempt {attempt + 1}/{retries})",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
            continue
        if response.status_code >= 400:
            raise FetchError(f"HTTP {response.status_code} from {url}")
        try:
            return response.json()
        except ValueError:
            raise FetchError(
                f"non-JSON response from {url}: {response.text[:200]!r}"
            )
    raise FetchError(
        f"{last} after {retries} attempts. This network may be sharing an IP "
        "with heavy API users. Try again later, or run the fetch from a "
        "different network."
    )


# ----------------------------------------------------------------- wikipedia


def wikipedia_featured_titles(session: requests.Session, count: int) -> List[str]:
    titles: List[str] = []
    cont: Optional[str] = None
    while len(titles) < count:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Featured articles",
            "cmlimit": "500",
            "cmnamespace": "1",  # Talk namespace holds the FA banners
            "format": "json",
        }
        if cont:
            params["cmcontinue"] = cont
        data = get_json(session, WIKI_API, params)
        for member in data.get("query", {}).get("categorymembers", []):
            title = member["title"]
            if title.startswith("Talk:"):
                titles.append(title[len("Talk:") :])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
    return titles[:count]


def wikipedia_extract(session: requests.Session, title: str) -> Optional[str]:
    """Plain-text extract at the pre-cutoff revision."""
    revs = get_json(
        session,
        WIKI_API,
        {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvlimit": "1",
            "rvstart": WIKI_CUTOFF,
            "rvdir": "older",
            "rvprop": "ids",
            "format": "json",
        },
    )
    pages = revs.get("query", {}).get("pages", {})
    revid = None
    for page in pages.values():
        for rev in page.get("revisions", []) or []:
            revid = rev.get("revid")
    if not revid:
        return None

    parsed = get_json(
        session,
        WIKI_API,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "revids": str(revid),
            "format": "json",
        },
    )
    for page in parsed.get("query", {}).get("pages", {}).values():
        text = page.get("extract")
        if text:
            return _strip_wiki_tail(text)
    return None


def _strip_wiki_tail(text: str) -> str:
    """Drop reference, note and bibliography sections; keep the prose body."""
    for marker in (
        "\nReferences\n",
        "\nNotes\n",
        "\nFurther reading\n",
        "\nExternal links\n",
        "\nSee also\n",
        "\nBibliography\n",
        "\nCitations\n",
    ):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    return text.strip()


def fetch_wikipedia(count: int, out: Path, delay: float) -> int:
    session = _session()
    print(f"listing Featured Articles ...", file=sys.stderr)
    titles = wikipedia_featured_titles(session, count * 2)
    print(f"  {len(titles)} candidate titles", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)

    saved = 0
    for title in titles:
        if saved >= count:
            break
        try:
            text = wikipedia_extract(session, title)
        except (requests.RequestException, FetchError) as exc:
            print(f"  skip {title}: {exc}", file=sys.stderr)
            continue
        if not text or len(text.split()) < 600:
            continue
        name = re.sub(r"[^\w.-]+", "_", title)[:80]
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
        saved += 1
        if saved % 10 == 0:
            print(f"  saved {saved}", file=sys.stderr)
        time.sleep(delay)
    print(f"saved {saved} documents to {out}", file=sys.stderr)
    return saved


# ----------------------------------------------------------------------- pmc


def fetch_pmc(count: int, out: Path, delay: float) -> int:
    session = _session()
    out.mkdir(parents=True, exist_ok=True)
    query = (
        "OPEN_ACCESS:y AND SRC:MED AND (FIRST_PDATE:[2016-01-01 TO 2021-12-31])"
    )
    saved = 0
    cursor = "*"
    while saved < count:
        data = get_json(
            session,
            EPMC_SEARCH,
            {
                "query": query,
                "resultType": "core",
                "format": "json",
                "pageSize": "100",
                "cursorMark": cursor,
            },
        )
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
        for record in results:
            if saved >= count:
                break
            pmcid = record.get("pmcid")
            if not pmcid:
                continue
            try:
                response = session.get(
                    EPMC_FULLTEXT.format(pmcid=pmcid), timeout=45
                )
                if response.status_code != 200:
                    continue
                text = _jats_body_text(response.text)
            except (requests.RequestException, FetchError):
                continue
            if not text or len(text.split()) < 600:
                continue
            (out / f"{pmcid}.txt").write_text(text, encoding="utf-8")
            saved += 1
            if saved % 10 == 0:
                print(f"  saved {saved}", file=sys.stderr)
            time.sleep(delay)
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    print(f"saved {saved} documents to {out}", file=sys.stderr)
    return saved


_TAG_RE = re.compile(r"<[^>]+>")
_BODY_RE = re.compile(r"<body\b[^>]*>(.*?)</body>", re.DOTALL | re.IGNORECASE)
_DROP_RE = re.compile(
    r"<(table-wrap|fig|xref|disp-formula|inline-formula|ref-list|graphic)\b.*?"
    r"</\1>|<(xref|graphic)\b[^>]*/>",
    re.DOTALL | re.IGNORECASE,
)
_PARA_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.DOTALL | re.IGNORECASE)


def _tidy_orphans(text: str) -> str:
    """Repair punctuation stranded by removed citation elements.

    Stripping <xref> tags leaves fragments like "disorders. , On the one hand",
    which the sentence splitter would treat as real punctuation and which would
    then show up as spurious comma density and short sentences.
    """
    text = re.sub(r"\s+([,;.)])", r"\1", text)
    text = re.sub(r"([.!?])\s*[,;]+", r"\1", text)
    text = re.sub(r"[,;]\s*([.!?])", r"\1", text)
    text = re.sub(r"\(\s*[,;]*\s*\)", "", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _jats_body_text(xml: str) -> str:
    """Extract body paragraphs from JATS XML.

    Tables, figures, formulas and cross-references are dropped: they fragment
    into pseudo-sentences and would corrupt the shape features exactly the way
    Markdown tables did.
    """
    body_match = _BODY_RE.search(xml)
    if not body_match:
        return ""
    body = _DROP_RE.sub(" ", body_match.group(1))
    paragraphs = []
    for raw in _PARA_RE.findall(body):
        text = _TAG_RE.sub("", raw)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\[\s*[\d,\s–-]*\s*\]", "", text)  # numeric citations
        text = _tidy_orphans(text)
        if len(text.split()) >= 25:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["wikipedia", "pmc"])
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="seconds between requests; be polite to public APIs",
    )
    args = parser.parse_args()

    try:
        if args.source == "wikipedia":
            saved = fetch_wikipedia(args.count, args.out, args.delay)
        else:
            saved = fetch_pmc(args.count, args.out, args.delay)
    except FetchError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 2

    if saved < 50:
        print(
            "warning: fewer than 50 documents. The covariance estimate needs "
            "300+ to be stable; research/10 recommends that as the floor.",
            file=sys.stderr,
        )
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
