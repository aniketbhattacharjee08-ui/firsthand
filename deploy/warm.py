"""Pre-download the hub checkpoints and run one rewrite so the first visitor
does not wait for a 4.5 GB download or a cold model load.

    .venv/bin/python deploy/warm.py            # download + one rewrite
    .venv/bin/python deploy/warm.py --no-run   # download only

Reads the same environment as the server (HUMANIZER_BASE_MODEL etc.).
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-run", action="store_true")
    args = p.parse_args()

    from humanizer.detectors import yardstick
    from humanizer.detectors.local import ModernDetector, PUBLISHED_DETECTORS

    print("yardstick:", yardstick.configured_name())
    t = time.time()
    det = ModernDetector(model_name=PUBLISHED_DETECTORS["surrogate"]["model"])
    if not det.available():
        print("surrogate unavailable:", det.unavailable_reason(), file=sys.stderr)
        return 1
    det.load()
    print("surrogate loaded in %.1fs" % (time.time() - t))

    try:
        from humanizer.humanize import llm as llm_mod
        from humanizer.humanize import pipeline as pipeline_mod
    except Exception as exc:  # noqa: BLE001
        print("LLM path unavailable:", exc, file=sys.stderr)
        return 0
    reason = pipeline_mod.pipeline_unavailable_reason()
    if reason:
        print("LLM path unavailable:", reason)
        return 0
    t = time.time()
    backend = llm_mod.backend_for("freeform")
    backend.load()
    print("base model %s loaded in %.1fs" % (llm_mod.FREEFORM_MODEL, time.time() - t))
    if args.no_run:
        return 0
    sample = (
        "In today's rapidly evolving landscape, organisations must leverage robust "
        "frameworks to navigate multifaceted challenges. Furthermore, stakeholders "
        "should delve into comprehensive strategies that underscore the importance of "
        "resilience. Consequently, a holistic approach is pivotal for sustained success."
    )
    t = time.time()
    result = pipeline_mod.humanize_llm(sample, config=pipeline_mod.PipelineConfig(n_candidates=2, time_budget_s=60))
    print("rewrite in %.1fs, changed=%s, p(AI) %s -> %s" % (
        time.time() - t, result.changed, result.before.get("ai_probability"), result.after.get("ai_probability")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
