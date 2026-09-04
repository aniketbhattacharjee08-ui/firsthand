"""Reference-distribution tests."""

import math

import numpy as np
import pytest

from humanizer.features import extract_features
from humanizer.reference import ReferenceDistribution, build_from_texts

WORDS = (
    "study result method analysis effect model data evidence theory sample "
    "participants measure outcome variable approach finding argument source"
).split()


def synthetic_document(seed: int, mean_len: float = 21.0, sd_len: float = 10.0) -> str:
    rng = np.random.default_rng(seed)
    paragraphs = []
    for _ in range(6):
        sentences = []
        for _ in range(int(rng.integers(3, 7))):
            n = int(max(4, rng.normal(mean_len, sd_len)))
            sentences.append(" ".join(rng.choice(WORDS, n)) + ".")
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)


@pytest.fixture(scope="module")
def reference() -> ReferenceDistribution:
    texts = [synthetic_document(i) for i in range(60)]
    return build_from_texts("synthetic", texts, source="test fixture", min_words=100)


class TestBuild:
    def test_documents_kept(self, reference):
        assert reference.n_documents == 60

    def test_has_features(self, reference):
        assert len(reference.feature_names) > 20
        assert "sent_len_cv" in reference.feature_names

    def test_min_words_filters(self):
        ref = build_from_texts(
            "tiny", ["short text here."] * 5 + [synthetic_document(1)], min_words=100
        )
        assert ref.n_documents == 1

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            build_from_texts("empty", [])

    def test_all_documents_filtered_raises(self):
        with pytest.raises(ValueError):
            build_from_texts("all-short", ["tiny."] * 5, min_words=1000)


class TestSummary:
    def test_summary_has_percentiles(self, reference):
        entry = reference.summary()["sent_len_cv"]
        for key in ("mean", "sd", "cv", "p10", "p50", "p90"):
            assert key in entry

    def test_percentiles_are_ordered(self, reference):
        entry = reference.summary()["sent_len_mean"]
        assert entry["p10"] <= entry["p50"] <= entry["p90"]


class TestDistance:
    def test_in_family_document_is_not_an_outlier(self, reference):
        feats = extract_features(synthetic_document(999))
        assert reference.score(feats)["percentile"] < 95.0

    def test_uniform_document_is_an_outlier(self, reference):
        flat = " ".join(
            ["The study shows a consistent effect across all measured groups."] * 40
        )
        score = reference.score(extract_features(flat))
        assert score["percentile"] >= 99.0

    def test_distance_is_finite_and_non_negative(self, reference):
        feats = extract_features(synthetic_document(500))
        d = reference.mahalanobis(feats)
        assert math.isfinite(d) and d >= 0

    def test_partial_coverage_still_scores(self, reference):
        # A single-paragraph document has no paragraph-level variance, so some
        # reference features are missing. It must still score, not return NaN.
        single = " ".join(
            [" ".join(WORDS[: (i % 12) + 5]) + "." for i in range(30)]
        )
        score = reference.score(extract_features(single))
        assert math.isfinite(score["distance"])
        assert score["coverage"] < 1.0

    def test_distances_are_finite(self, reference):
        # Constant features previously made the covariance singular and put inf
        # in the inverse. Underflow is benign and not checked here.
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            distances = reference.reference_mahalanobis()
        assert np.isfinite(distances).all()

    def test_zscores_identify_deviant_features(self, reference):
        flat = " ".join(["Short and even sentences repeat here again now."] * 40)
        zs = reference.zscores(extract_features(flat))
        assert zs
        ranked = sorted(zs.items(), key=lambda kv: -abs(kv[1]))
        # The document is degenerate on both shape and lexical diversity, so
        # only require that the top deviations are large and that shape appears.
        assert abs(ranked[0][1]) > 5
        top_names = {name for name, _ in ranked[:6]}
        assert any(
            key in name
            for name in top_names
            for key in ("sent", "para", "mtld", "hdd", "ttr")
        )


class TestSampling:
    def test_sample_profile_matches_a_real_document(self, reference):
        rng = np.random.default_rng(0)
        profile = reference.sample_profile(rng)
        assert set(profile) == set(reference.feature_names)
        # Without jitter the profile is exactly one reference row, which is how
        # covariance is preserved.
        rows = {tuple(np.round(r, 9)) for r in reference.matrix}
        drawn = tuple(np.round([profile[n] for n in reference.feature_names], 9))
        assert drawn in rows

    def test_jitter_moves_the_profile(self, reference):
        rng = np.random.default_rng(0)
        base = reference.sample_profile(np.random.default_rng(0))
        jittered = reference.sample_profile(rng, jitter=0.5)
        assert base != jittered

    def test_persona_offsets_scale_with_icc(self, reference):
        # High-ICC punctuation should vary more between personas than low-ICC
        # features, since the offset is sd * sqrt(rho).
        rng = np.random.default_rng(1)
        draws = [reference.sample_persona(rng) for _ in range(400)]
        if "comma_per_1k" in reference.feature_names and "hedge_per_1k" in reference.feature_names:
            comma = np.std([d["comma_per_1k"] for d in draws])
            hedge = np.std([d["hedge_per_1k"] for d in draws])
            comma_sd = reference.sd[reference.feature_names.index("comma_per_1k")]
            hedge_sd = reference.sd[reference.feature_names.index("hedge_per_1k")]
            if comma_sd > 0 and hedge_sd > 0:
                assert (comma / comma_sd) > (hedge / hedge_sd)


class TestPersistence:
    def test_roundtrip(self, reference, tmp_path):
        path = tmp_path / "ref.json"
        reference.to_json(path)
        loaded = ReferenceDistribution.from_json(path)
        assert loaded.genre == reference.genre
        assert loaded.feature_names == reference.feature_names
        assert np.allclose(loaded.matrix, reference.matrix)

    def test_roundtrip_preserves_distance(self, reference, tmp_path):
        path = tmp_path / "ref.json"
        reference.to_json(path)
        loaded = ReferenceDistribution.from_json(path)
        feats = extract_features(synthetic_document(321))
        assert loaded.mahalanobis(feats) == pytest.approx(
            reference.mahalanobis(feats), rel=1e-9
        )
