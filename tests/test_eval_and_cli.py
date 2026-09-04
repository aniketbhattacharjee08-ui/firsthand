"""Report, findings and CLI tests."""

import json

import pytest

from humanizer.clean import clean_markup, looks_like_markup
from humanizer.cli import main
from humanizer.detectors import Detector, DetectorResult
from humanizer.eval import analyze, build_findings
from humanizer.features import extract_features

AI_LIKE = (
    "In today's rapidly evolving landscape, it is important to note that this "
    "plays a pivotal role in outcomes. Furthermore, the intricate tapestry of "
    "technology underscores the importance of understanding. Moreover, we must "
    "delve into the multifaceted challenges here. Additionally, robust "
    "frameworks are crucial for success now. Consequently, stakeholders should "
    "leverage comprehensive strategies today. "
) * 4


class TestClean:
    def test_detects_markup(self):
        assert looks_like_markup("# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |")

    def test_plain_prose_is_not_markup(self):
        assert not looks_like_markup("A plain sentence. Another one follows.")

    def test_strips_tables(self):
        cleaned = clean_markup("Text here.\n\n| a | b |\n|---|---|\n| 1 | 2 |")
        assert "|" not in cleaned

    def test_strips_headings_and_keeps_prose(self):
        cleaned = clean_markup("## Title\n\nReal prose stays here.")
        assert "Title" not in cleaned
        assert "Real prose stays here." in cleaned

    def test_keeps_link_text(self):
        assert clean_markup("See [the docs](http://x.com) now.") == "See the docs now."

    def test_strips_code_fences(self):
        assert "print" not in clean_markup("Before.\n\n```\nprint(1)\n```\n\nAfter.")

    def test_tables_would_corrupt_shape_features(self):
        """The reason cleaning is mandatory, not optional."""
        table = "Real prose sentence here today.\n\n" + "\n".join(
            f"| row {i} | value |" for i in range(40)
        )
        import math

        raw = extract_features(table)
        cleaned = extract_features(clean_markup(table))
        # Pipe rows carry no terminal punctuation, so uncleaned they collapse
        # into one enormous pseudo-sentence and drag the CV to 1.25, twice the
        # top of the human band.
        assert raw["sent_len_cv"] > 0.9
        assert raw["sent_len_max"] > 50
        # Cleaned, only the real prose sentence survives.
        assert cleaned["n_sentences"] == 1
        assert cleaned["sent_len_max"] <= 5


class TestFindings:
    def test_ai_text_produces_findings(self):
        feats = extract_features(AI_LIKE)
        findings = build_findings(feats, {})
        assert findings
        codes = {f.code for f in findings}
        assert "ai_vocabulary" in codes
        assert "formal_connectives" in codes

    def test_findings_sorted_by_severity(self):
        findings = build_findings(extract_features(AI_LIKE), {})
        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        severities = [order[f.severity] for f in findings]
        assert severities == sorted(severities)

    def test_connective_finding_reports_negative_grade_cost(self):
        """The asymmetry that makes the product possible."""
        findings = build_findings(extract_features(AI_LIKE), {})
        match = next(f for f in findings if f.code == "formal_connectives")
        assert "raises" in match.grade_cost

    def test_overshoot_is_flagged_not_rewarded(self):
        # Alternating short and 70-word sentences: CV far above the band.
        # Capitalised openers, because the splitter deliberately declines to
        # break before a lowercase continuation after a short token.
        sentences = [
            ("Short one." if i % 2 else "Word " + " ".join(["word"] * 69) + ".")
            for i in range(30)
        ]
        feats = extract_features(" ".join(sentences))
        from humanizer.features import band_report

        findings = build_findings(feats, band_report(feats))
        assert any(f.code == "overshot_variance" for f in findings)

    def test_opinion_marker_is_high_severity(self):
        feats = extract_features("I think this is correct and useful. " * 12)
        match = next(
            f for f in build_findings(feats, {}) if f.code == "opinion_markers"
        )
        assert match.severity == "high"
        assert match.grade_cost == "high"

    def test_clean_prose_has_few_findings(self):
        prose = (
            "The archive closed at four, so I had ninety minutes with the "
            "ledger. Most of it was unreadable. Water damage had taken the "
            "left margin of every third page, which is where the clerk "
            "recorded dates. What survived was a column of names and a column "
            "of sums. Between them ran a narrow gutter where someone had "
            "later pencilled corrections that disagree with the original "
            "entries by amounts too small to be transcription errors. "
        ) * 3
        codes = {f.code for f in build_findings(extract_features(prose), {})}
        assert "ai_vocabulary" not in codes
        assert "formal_connectives" not in codes


class TestReport:
    def test_render_is_a_string(self):
        assert isinstance(analyze(AI_LIKE).render(), str)

    def test_json_roundtrip(self):
        payload = json.loads(analyze(AI_LIKE).to_json())
        assert "features" in payload and "findings" in payload

    def test_detector_included(self):
        """`analyze()` is generic over detectors; a stub proves the plumbing.

        A stub rather than a real detector on purpose. The real ones are
        published checkpoints of 130MB to 1.7GB and this test is about
        `analyze()` calling `score()` and filing the result under
        `detector.name`, not about any model.
        """

        class Stub(Detector):
            name = "stub"

            def score(self, text):
                return DetectorResult(detector=self.name, ai_probability=0.5,
                                      label="ai", confidence="low")

        report = analyze(AI_LIKE, detectors=[Stub()])
        assert "stub" in report.detectors
        assert report.detectors["stub"]["ai_probability"] == 0.5

    def test_failing_detector_is_recorded_not_raised(self):
        """A missing 1.7GB download must degrade the report, not kill it."""

        class Broken(Detector):
            name = "broken"

            def score(self, text):
                raise RuntimeError("no key")

        report = analyze(AI_LIKE, detectors=[Broken()])
        assert "error" in report.detectors["broken"]

    def test_auto_clean_applied(self):
        markup = "# Title\n\n" + "\n".join(f"| a{i} | b |" for i in range(30))
        # Must not crash and must not treat table rows as sentences.
        analyze(markup)


class TestCLI:
    def test_features_command(self, tmp_path, capsys):
        path = tmp_path / "in.txt"
        path.write_text(AI_LIKE)
        assert main(["features", str(path), "--json"]) == 0
        assert "sent_len_cv" in json.loads(capsys.readouterr().out)

    def test_analyze_command(self, tmp_path, capsys):
        path = tmp_path / "in.txt"
        path.write_text(AI_LIKE)
        assert main(["analyze", str(path)]) == 0
        assert "Findings" in capsys.readouterr().out

    def test_analyze_json(self, tmp_path, capsys):
        path = tmp_path / "in.txt"
        path.write_text(AI_LIKE)
        assert main(["analyze", str(path), "--json"]) == 0
        assert "features" in json.loads(capsys.readouterr().out)

    def test_plan_command(self, capsys):
        assert main(["plan", "--mu", "0.6", "--target", "0.99"]) == 0
        assert "never" in capsys.readouterr().out  # rho=0.5 is unreachable

    def test_build_reference_command(self, tmp_path, capsys):
        import numpy as np

        rng = np.random.default_rng(0)
        words = "study result method analysis effect model data evidence".split()
        for i in range(20):
            r = np.random.default_rng(i)
            doc = "\n\n".join(
                " ".join(
                    " ".join(r.choice(words, int(max(4, r.normal(20, 9))))) + "."
                    for _ in range(5)
                )
                for _ in range(6)
            )
            (tmp_path / f"d{i}.txt").write_text(doc)
        out = tmp_path / "ref.json"
        code = main(
            [
                "build-reference",
                "synthetic",
                str(tmp_path / "*.txt"),
                "--out",
                str(out),
                "--min-words",
                "100",
            ]
        )
        assert code == 0 and out.exists()

    def test_build_reference_no_matches(self, tmp_path, capsys):
        assert main(
            ["build-reference", "g", str(tmp_path / "none*.txt"), "--out", "x.json"]
        ) == 1

    def test_calibrate_command(self, tmp_path, capsys):
        rows = [
            {"local_human_score": i / 100, "gptzero_human": i > 50}
            for i in range(100)
        ]
        path = tmp_path / "cal.json"
        path.write_text(json.dumps(rows))
        assert main(["calibrate", "--scores", str(path)]) == 0
        assert "Calibration" in capsys.readouterr().out
