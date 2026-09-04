"""Command-line interface.

    humanizer analyze FILE [--reference data/reference/academic.json] [--json]
    humanizer features FILE
    humanizer build-reference GENRE PATTERN --out FILE
    humanizer calibrate --scores FILE
    humanizer plan --mu 0.6 --target 0.99 --rho 0.2
    humanizer serve --port 8000 --reference data/reference/academic.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .detectors import HeuristicDetector, pass_rate, required_n
from .eval import analyze
from .reference import ReferenceDistribution, build_from_texts


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _load_reference(path: Optional[str]) -> Optional[ReferenceDistribution]:
    if not path:
        return None
    return ReferenceDistribution.from_json(Path(path))


def cmd_analyze(args: argparse.Namespace) -> int:
    text = _read(args.file)
    detectors = [] if args.no_detectors else [HeuristicDetector()]
    if args.gptzero:
        from .detectors import GPTZeroClient

        client = GPTZeroClient()
        if not client.available():
            print(
                "warning: GPTZERO_API_KEY is not set, skipping GPTZero",
                file=sys.stderr,
            )
        else:
            detectors.append(client)
    report = analyze(text, reference=_load_reference(args.reference), detectors=detectors)
    print(report.to_json() if args.json else report.render())
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    from .clean import clean_markup, looks_like_markup
    from .features import extract_features

    text = _read(args.file)
    if looks_like_markup(text):
        text = clean_markup(text)
    feats = extract_features(text)
    if args.json:
        print(json.dumps(feats, indent=2))
    else:
        for name in sorted(feats):
            print(f"{name:<36} {feats[name]:.4f}")
    return 0


def cmd_build_reference(args: argparse.Namespace) -> int:
    paths: List[str] = []
    for pattern in args.patterns:
        paths.extend(sorted(glob.glob(pattern, recursive=True)))
    if not paths:
        print("no files matched", file=sys.stderr)
        return 1

    texts, ids = [], []
    for path in paths:
        try:
            texts.append(Path(path).read_text(encoding="utf-8", errors="replace"))
            ids.append(Path(path).name)
        except OSError as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)

    ref = build_from_texts(
        args.genre, texts, doc_ids=ids, source=", ".join(args.patterns),
        min_words=args.min_words,
    )
    ref.to_json(Path(args.out))
    kept, seen = ref.n_documents, len(texts)
    print(
        f"built '{args.genre}' from {kept}/{seen} documents "
        f"({len(ref.feature_names)} features) -> {args.out}"
    )
    if kept < 50:
        print(
            f"warning: {kept} documents is thin. Aim for 300+ so the "
            "covariance estimate is stable.",
            file=sys.stderr,
        )
    dists = ref.reference_mahalanobis()
    print(
        f"reference distance: median {dists.mean():.2f}, "
        f"90th percentile {sorted(dists)[int(0.9 * len(dists)) - 1]:.2f}"
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Calibrate a local score against GPTZero verdicts.

    Input is JSON: a list of {"local_human_score": float, "gptzero_human": bool}.
    """
    from .detectors import calibrate

    rows = json.loads(_read(args.scores))
    scores = [float(r["local_human_score"]) for r in rows]
    labels = [bool(r["gptzero_human"]) for r in rows]
    cal = calibrate(scores, labels)
    print(cal.summary())
    chosen = cal.select_threshold(args.target)
    print()
    if chosen:
        print(
            f"Operating point for {args.target:.0%}: accept at local score "
            f">= {chosen.threshold:.2f} "
            f"(measured {chosen.rate:.1%}, lower bound {chosen.lower:.1%}, "
            f"n = {chosen.n})"
        )
    else:
        print(
            f"No threshold reaches {args.target:.0%} at the 95% lower bound. "
            "Collect more samples or improve the local detector."
        )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Best-of-N planning under candidate correlation."""
    print(f"Per-candidate success mu = {args.mu:.2f}, target = {args.target:.1%}")
    print(f"{'rho':>6} {'N needed':>10} {'pass @ N=8':>12} {'pass @ N=32':>12}")
    for rho in (0.0, 0.1, 0.2, 0.3, 0.5):
        n = required_n(args.mu, args.target, rho)
        print(
            f"{rho:>6.1f} {str(n) if n else '  never':>10} "
            f"{pass_rate(args.mu, 8, rho):>12.4f} "
            f"{pass_rate(args.mu, 32, rho):>12.4f}"
        )
    print()
    print(
        "Correlation between candidates, not raw N, is the binding constraint. "
        "Diversify plans and base models before spending more compute."
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve the HTTP API and, when web/ exists, the static frontend."""
    try:
        from .api import run as run_server
    except ImportError as exc:
        print(
            f"the API needs the 'api' extra: pip install -e '.[api]'  ({exc})",
            file=sys.stderr,
        )
        return 1

    print(f"humanizer API on http://{args.host}:{args.port}")
    if args.reference:
        print(f"default reference: {args.reference}")
    run_server(host=args.host, port=args.port, reference=args.reference)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanizer", description="Stage 0 analysis toolkit."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="full report for one document")
    p.add_argument("file", help="path to a text file, or - for stdin")
    p.add_argument("--reference", help="path to a reference distribution JSON")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--gptzero", action="store_true", help="also query GPTZero")
    p.add_argument("--no-detectors", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("features", help="dump the raw feature vector")
    p.add_argument("file")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("build-reference", help="build a reference distribution")
    p.add_argument("genre")
    p.add_argument("patterns", nargs="+", help="glob patterns for input files")
    p.add_argument("--out", required=True)
    p.add_argument("--min-words", type=int, default=300)
    p.set_defaults(func=cmd_build_reference)

    p = sub.add_parser("calibrate", help="map local scores to GPTZero verdicts")
    p.add_argument("--scores", required=True, help="JSON file of paired results")
    p.add_argument("--target", type=float, default=0.95)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("plan", help="best-of-N planning under correlation")
    p.add_argument("--mu", type=float, default=0.6)
    p.add_argument("--target", type=float, default=0.99)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("serve", help="run the HTTP API and web frontend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--reference",
        help="default reference distribution (name or path) for /api/analyze",
    )
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
