"""Your voice: the fingerprint, the exemplars, the gate-first ranking, the routes.

The rules under test, in the order the feature makes them:

1. The detector gate is hard. A voice chooses among human-reading
   candidates and never promotes an AI-reading one.
2. Samples that read as AI are refused at upload.
3. Model output is never added to the sample pool: a thumbs up goes to a
   separate pile, used as an exemplar only below three real samples.
4. A guest has no voice; one user cannot see or vote on another's.
"""

from __future__ import annotations

import pytest

from humanizer.humanize import voice as voice_mod

HUMAN = (
    "The library was shut on Tuesday, so I walked. It rained; I did not mind. "
    "Nobody tells you how long the road is until you are on it, and by then the "
    "question is moot. We argued about whether the bridge counted. It did not. "
    "In 2019 the council spent 12,000 dollars on the sign alone, which Mr Harding "
    "called a scandal, and he was not wrong. The sign is still there. I pass it "
    "most weeks and think about the meeting where they voted, four to three, with "
    "one abstention nobody has ever explained. That is local government for you. "
    "You get the sign you deserve, and then you get the bill."
)
HUMAN_2 = (
    "My grandmother kept bees behind the shed, which the neighbours hated. She "
    "never wore gloves; she said the bees knew her. I doubt it, but she was stung "
    "less than I was, and I stood twenty feet back. In August 1987 the hive swarmed "
    "into the Parkers' apple tree and stayed there for a week, and Mr Parker, who "
    "was a careful man, called the fire brigade twice. They came once. The second "
    "time the dispatcher told him to call a beekeeper, so he called my grandmother, "
    "and she walked over in her apron and brought the swarm home in a cardboard box. "
    "That is the whole story. People still tell it at funerals, which is how you "
    "know it is true, or at least that it is ours."
)
GENERIC = (
    "Furthermore, it is important to note that the implementation of comprehensive "
    "strategies facilitates optimal outcomes across diverse stakeholder groups. "
    "Moreover, organizations must prioritize the integration of innovative "
    "approaches in order to enhance efficiency and ensure sustainable growth. "
    "Additionally, careful consideration of these factors is essential for success."
)


# ------------------------------------------------------------ the fingerprint


class TestProfile:
    def test_builds_from_samples_and_reads_the_author_closer_than_generic_prose(self):
        p = voice_mod.build_profile([HUMAN, HUMAN_2])
        assert p.n_samples == 2
        assert p.n_words > 150
        assert "sent_len_cv" in p.means and "comma_per_1k" in p.means
        assert p.trigrams
        own = voice_mod.profile_distance(p, HUMAN)
        other = voice_mod.profile_distance(p, GENERIC)
        assert 0.0 <= own < other <= 1.0

    def test_round_trips_through_json_shape(self):
        p = voice_mod.build_profile([HUMAN])
        again = voice_mod.VoiceProfile.from_dict(p.as_dict())
        assert again.means == p.means and again.trigrams == p.trigrams
        assert not again.empty

    def test_empty_profile_has_zero_distance_and_no_traits(self):
        p = voice_mod.build_profile([])
        assert p.empty
        assert voice_mod.profile_distance(p, HUMAN) == 0.0
        assert voice_mod.describe_profile(p) == []

    def test_traits_are_plain_words_with_no_dashes(self):
        rows = voice_mod.describe_profile(voice_mod.build_profile([HUMAN, HUMAN_2]))
        labels = {r["label"] for r in rows}
        assert "Sentence length" in labels and "Punctuation" in labels
        for r in rows:
            for key in ("label", "value", "note"):
                assert "—" not in r[key] and "–" not in r[key]


class TestNotes:
    def test_bullets_carry_only_sentences_with_particulars(self):
        notes = voice_mod.notes_from(HUMAN)
        lines = notes.splitlines()
        assert all(line.startswith("- ") for line in lines)
        assert any("12,000" in line for line in lines)
        assert not any(line == "- It rained; I did not mind." for line in lines)

    def test_no_particulars_gives_the_stock_line(self):
        assert voice_mod.notes_from("it was fine. we went home. nothing happened.").startswith("- No particulars")


class TestFlatten:
    def test_prompt_is_a_base_model_pattern_that_ends_at_the_open_draft_slot(self):
        from humanizer.humanize.exemplars import FREEFORM_SHOTS

        p = voice_mod.flatten_prompt(HUMAN)
        assert p.kind == "text"
        assert p.text.endswith("HUMAN:\n" + HUMAN + "\n\nDRAFT:\n")
        # The stock pairs appear reversed: the human rewrite first, its draft after.
        _t, draft, _n, rewrite = FREEFORM_SHOTS[0]
        assert p.text.index(rewrite) < p.text.index(draft)

    def test_cleaning_cuts_at_the_patterns_next_label(self):
        twin = GENERIC + "\n\nHUMAN:\nsomething else the model went on to write"
        assert voice_mod.clean_flattened(twin, HUMAN) == GENERIC

    def test_cleaning_refuses_echo_refusal_and_wild_lengths(self):
        assert voice_mod.clean_flattened(HUMAN, HUMAN) is None
        assert voice_mod.clean_flattened("I'm sorry, I cannot do that.", HUMAN) is None
        assert voice_mod.clean_flattened("Short.", HUMAN) is None
        assert voice_mod.clean_flattened(GENERIC + " " + GENERIC, HUMAN) is not None


class TestShots:
    def _samples(self, n):
        out = []
        for i in range(n):
            out.append({"id": "s_%d" % i, "title": "t%d" % i, "text": HUMAN if i % 2 else HUMAN_2, "draft": GENERIC, "notes": "- x", "weight": 1.0})
        return out

    def test_real_samples_fill_every_slot_and_the_liked_pile_is_ignored(self):
        liked = [{"id": "l_1", "original": GENERIC, "text": HUMAN}]
        shots, ids = voice_mod.select_shots(self._samples(4), HUMAN, liked=liked)
        assert len(shots) == 3
        assert all(i.startswith("s_") for i in ids)

    def test_below_three_real_samples_one_liked_rewrite_fills_one_slot_only(self):
        liked = [{"id": "l_1", "original": GENERIC, "text": HUMAN}, {"id": "l_2", "original": GENERIC, "text": HUMAN_2}]
        shots, ids = voice_mod.select_shots(self._samples(1), HUMAN, liked=liked)
        assert len(shots) == 2
        assert sum(1 for i in ids if i.startswith("l_")) == 1

    def test_samples_without_a_twin_are_not_exemplars(self):
        samples = self._samples(2)
        samples[0]["draft"] = ""
        shots, ids = voice_mod.select_shots(samples, HUMAN)
        assert ids == ("s_1",)

    def test_nothing_gives_nothing(self):
        assert voice_mod.select_shots([], HUMAN) == ((), ())


# ------------------------------------------------------------- the pipeline


class TestPipelineRanking:
    """Rule one: the gate is hard."""

    def _cand(self, index, p, text):
        from humanizer.humanize.pipeline import Candidate

        return Candidate(paragraph_index=0, index=index, persona="", constraint="", temperature=0.7, ai_probability=p, scrubbed=text, raw=text)

    def test_voice_chooses_among_passers_only(self):
        from humanizer.humanize import pipeline as pipeline_mod

        profile = voice_mod.build_profile([HUMAN, HUMAN_2])
        cfg = pipeline_mod.PipelineConfig(voice=voice_mod.VoiceSpec(profile=profile, strength="strong"), pass_threshold=0.15)
        # The closest-to-voice candidate reads as AI; the two passers do not.
        cands = [self._cand(0, 0.9, HUMAN), self._cand(1, 0.10, GENERIC), self._cand(2, 0.12, HUMAN_2)]
        best = pipeline_mod._select_scored(cands, cfg)
        assert best.index == 2, "the human-reading candidate closest to the voice wins"
        assert cands[2].quality["voice_distance"] < cands[1].quality["voice_distance"]

    def test_without_a_passer_the_voice_is_ignored(self):
        from humanizer.humanize import pipeline as pipeline_mod

        profile = voice_mod.build_profile([HUMAN, HUMAN_2])
        cfg = pipeline_mod.PipelineConfig(voice=voice_mod.VoiceSpec(profile=profile, strength="strong"), pass_threshold=0.15)
        cands = [self._cand(0, 0.9, HUMAN), self._cand(1, 0.4, GENERIC)]
        assert pipeline_mod._select_scored(cands, cfg).index == 1

    def test_no_voice_is_the_plain_argmin(self):
        from humanizer.humanize import pipeline as pipeline_mod

        cfg = pipeline_mod.PipelineConfig()
        cands = [self._cand(0, 0.3, HUMAN), self._cand(1, 0.2, GENERIC)]
        assert pipeline_mod._select_scored(cands, cfg).index == 1


class TestPipelineShots:
    def test_the_users_samples_are_the_exemplars_the_base_model_sees(self):
        from humanizer.humanize import llm as llm_mod
        from humanizer.humanize import pipeline as pipeline_mod

        seen = []

        def echo(req):
            seen.append(req.prompt.text)
            return HUMAN_2

        backend = llm_mod.EchoBackend(echo)
        shots = ((HUMAN[:40], GENERIC, "- x", HUMAN),)
        spec = voice_mod.VoiceSpec(profile=voice_mod.build_profile([HUMAN]), shots=shots, sample_ids=("s_1",), run_id="r_1")
        cfg = pipeline_mod.PipelineConfig(n_candidates=2, rounds=1, style="freeform", voice=spec, time_budget_s=60)
        result = pipeline_mod.humanize_llm(GENERIC, config=cfg, backend=backend, detector=False)
        assert seen and all(HUMAN in text for text in seen)
        assert all(GENERIC in text for text in seen)
        v = result.summary["voice"]
        assert v["used"] is True and v["run_id"] == "r_1" and v["sample_ids"] == ["s_1"] and v["shots"] == 1

    def test_an_unusable_voice_reports_why(self):
        from humanizer.humanize import llm as llm_mod
        from humanizer.humanize import pipeline as pipeline_mod

        backend = llm_mod.EchoBackend(lambda req: HUMAN_2)
        cfg = pipeline_mod.PipelineConfig(n_candidates=1, rounds=1, style="freeform", voice=voice_mod.VoiceSpec(reason="no_samples"))
        result = pipeline_mod.humanize_llm(GENERIC, config=cfg, backend=backend, detector=False)
        assert result.summary["voice"] == {"used": False, "reason": "no_samples"}

    def test_no_voice_means_no_voice_block(self):
        from humanizer.humanize import llm as llm_mod
        from humanizer.humanize import pipeline as pipeline_mod

        cfg = pipeline_mod.PipelineConfig(n_candidates=1, rounds=1, style="freeform")
        result = pipeline_mod.humanize_llm(GENERIC, config=cfg, backend=llm_mod.EchoBackend(lambda r: HUMAN), detector=False)
        assert result.summary["voice"] is None


# ----------------------------------------------------------------- the API


pytest.importorskip("fastapi", reason="the api extra is not installed")
pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api import voice as api_voice  # noqa: E402
from humanizer.api.server import create_app  # noqa: E402
from humanizer.humanize import llm as llm_mod  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    # No free detector in the test environment: the upload check is skipped
    # and `ai_probability` is null, which the routes treat as "not refused".
    monkeypatch.setattr(api_voice, "_sample_ai_probability", lambda text: None)
    built = create_app(auth_db=tmp_path / "auth.sqlite", voice_db=tmp_path / "voice.sqlite")
    # The instruct model is an echo that returns a generic twin about as
    # long as the sample it was asked to flatten.
    built.state.voice_preparer.backend_factory = lambda: llm_mod.EchoBackend(_twin_for)
    return built


def _twin_for(req):
    sample = req.prompt.text.rsplit("HUMAN:\n", 1)[-1].split("\n\nDRAFT:")[0]
    n = len(sample.split())
    out = GENERIC
    while len(out.split()) < 0.7 * n:
        out += " " + GENERIC
    return out


def _signup(c, email):
    r = c.post("/api/auth/signup", json={"email": email, "password": "password123", "name": "A"})
    assert r.status_code == 201, r.text


def _prepared(app):
    """Wait for the background preparer to drain its queue."""
    app.state.voice_preparer._queue.join()


class TestVoiceRoutes:
    def test_a_visitor_sees_signed_out_and_cannot_write(self, app):
        with TestClient(app) as c:
            body = c.get("/api/voice").json()
            assert body["signed_in"] is False and body["samples"] == []
            assert c.post("/api/voice/samples", json={"text": HUMAN}).status_code == 401
            assert c.post("/api/voice/settings", json={"enabled": True}).status_code == 401
            assert c.post("/api/voice/feedback", json={"vote": "up", "original": GENERIC, "text": HUMAN}).status_code == 401

    def test_the_page_is_served(self, app):
        with TestClient(app) as c:
            r = c.get("/voice")
            assert r.status_code == 200 and "Your voice" in r.text

    def test_upload_prepare_profile_and_the_switch(self, app):
        with TestClient(app) as c:
            _signup(c, "one@example.com")
            short = c.post("/api/voice/samples", json={"text": "Too short."})
            assert short.status_code == 422 and short.json()["error"] == "too_short"
            r = c.post("/api/voice/samples", json={"title": "Essay", "text": HUMAN})
            assert r.status_code == 201, r.text
            sample = r.json()["sample"]
            assert sample["status"] == "preparing" and sample["n_words"] > 80
            assert "text" not in sample, "the list never echoes the full text"
            dup = c.post("/api/voice/samples", json={"text": HUMAN})
            assert dup.status_code == 422 and dup.json()["error"] == "duplicate"
            _prepared(app)
            body = c.get("/api/voice").json()
            assert body["signed_in"] is True and body["ready"] == 1
            assert body["samples"][0]["status"] == "ready"
            assert body["profile"]["n_samples"] == 1 and body["profile"]["traits"]
            assert body["enabled"] is True and body["strength"] == "balanced"
            s = c.post("/api/voice/settings", json={"enabled": False, "strength": "strong"}).json()
            assert s == {"enabled": False, "strength": "strong"}
            assert c.post("/api/voice/settings", json={"strength": "loud"}).status_code == 422

    def test_an_ai_written_sample_is_refused(self, app, monkeypatch):
        monkeypatch.setattr(api_voice, "_sample_ai_probability", lambda text: 0.93)
        with TestClient(app) as c:
            _signup(c, "two@example.com")
            r = c.post("/api/voice/samples", json={"text": GENERIC + " " + GENERIC})
            assert r.status_code == 422
            assert r.json()["error"] == "reads_as_ai" and "93" in r.json()["detail"]
            assert c.get("/api/voice").json()["samples"] == []

    def test_without_a_model_the_sample_fails_with_a_reason_but_the_profile_works(self, app):
        app.state.voice_preparer.backend_factory = None
        with TestClient(app) as c:
            _signup(c, "three@example.com")
            r = c.post("/api/voice/samples", json={"text": HUMAN})
            assert r.status_code == 201
            assert r.json()["sample"]["status"] == "failed" and "No language model" in r.json()["sample"]["detail"]
            body = c.get("/api/voice").json()
            assert body["ready"] == 0 and body["profile"] is not None and body["llm_available"] is False

    def test_feedback_moves_weights_and_keeps_liked_rewrites_apart(self, app):
        with TestClient(app) as c:
            _signup(c, "four@example.com")
            sid = c.post("/api/voice/samples", json={"text": HUMAN}).json()["sample"]["id"]
            _prepared(app)
            r = c.post("/api/voice/feedback", json={"vote": "up", "original": GENERIC, "text": HUMAN_2, "run_id": "r_abc", "sample_ids": [sid, "s_notmine"]})
            assert r.status_code == 200 and r.json()["liked_count"] == 1
            body = c.get("/api/voice").json()
            assert body["samples"][0]["weight"] == 1.25
            assert len(body["liked"]) == 1 and len(body["samples"]) == 1, "the liked rewrite is not a sample"
            # A second vote on the same run replaces the first.
            r = c.post("/api/voice/feedback", json={"vote": "down", "original": GENERIC, "text": HUMAN_2, "run_id": "r_abc", "sample_ids": [sid]})
            assert r.json()["liked_count"] == 0
            body = c.get("/api/voice").json()
            assert body["samples"][0]["weight"] == pytest.approx(0.65)
            assert c.post("/api/voice/feedback", json={"vote": "sideways"}).status_code == 422
            lid = c.post("/api/voice/feedback", json={"vote": "up", "original": GENERIC, "text": HUMAN_2, "run_id": "r_def"}).json()
            liked_id = c.get("/api/voice").json()["liked"][0]["id"]
            assert c.delete("/api/voice/liked/" + liked_id).json() == {"deleted": True}
            assert c.delete("/api/voice/samples/" + sid).json() == {"deleted": True}
            assert c.get("/api/voice").json()["profile"] is None

    def test_one_user_cannot_touch_anothers_samples(self, app, tmp_path):
        with TestClient(app) as a:
            _signup(a, "a@example.com")
            sid = a.post("/api/voice/samples", json={"text": HUMAN}).json()["sample"]["id"]
        with TestClient(app) as b:
            _signup(b, "b@example.com")
            assert b.get("/api/voice").json()["samples"] == []
            assert b.delete("/api/voice/samples/" + sid).status_code == 404
            assert b.post("/api/voice/samples/" + sid + "/prepare").status_code == 404
            assert b.delete("/api/voice/samples/s_%00bad").status_code in (404, 422)

    def test_a_run_with_voice_on_carries_the_users_spec_into_the_config(self, app, monkeypatch):
        from humanizer.humanize import pipeline as pipeline_mod

        captured = {}

        real = pipeline_mod.humanize_llm

        def fake_humanize(text, config=None, detector=None, backend=None):
            captured["config"] = config
            return real(text, config=config, backend=llm_mod.EchoBackend(lambda r: HUMAN_2), detector=False)

        monkeypatch.setattr(pipeline_mod, "humanize_llm", fake_humanize)
        with TestClient(app) as c:
            # Signed out: the run proceeds and says the voice was not used.
            r = c.post("/api/humanize/llm", json={"text": GENERIC, "voice": True, "n_candidates": 1, "rounds": 1})
            assert r.status_code == 200, r.text
            assert r.json()["summary"]["voice"] == {"used": False, "reason": "login_required"}
            _signup(c, "five@example.com")
            sid = c.post("/api/voice/samples", json={"text": HUMAN}).json()["sample"]["id"]
            _prepared(app)
            r = c.post("/api/humanize/llm", json={"text": GENERIC, "voice": True, "n_candidates": 1, "rounds": 1})
            assert r.status_code == 200, r.text
            v = r.json()["summary"]["voice"]
            assert v["used"] is True and v["sample_ids"] == [sid] and v["run_id"].startswith("r_")
            spec = captured["config"].voice
            assert spec.usable and spec.profile is not None and callable(spec.shots_for)
            assert c.get("/api/voice").json()["samples"][0]["uses"] == 1
            # Off by default: no voice block at all.
            r = c.post("/api/humanize/llm", json={"text": GENERIC, "n_candidates": 1, "rounds": 1})
            assert r.json()["summary"]["voice"] is None


# --------------------------------------------------- the review's fixes


class TestReviewFixes:
    def test_long_uploads_split_into_paragraph_sized_parts(self):
        long = "\n\n".join([HUMAN, HUMAN_2] * 4)  # about 900 words
        parts = voice_mod.split_sample(long, part_words=300)
        assert len(parts) >= 3
        assert all(len(voice_mod.words(p)) <= 300 + 10 for p in parts[:-1])
        assert all(len(voice_mod.words(p)) >= voice_mod.MIN_SAMPLE_WORDS for p in parts)
        assert voice_mod.split_sample(HUMAN) == [HUMAN]
        # One huge paragraph is cut at sentence ends rather than mid-sentence.
        one = " ".join([HUMAN] * 6)
        for p in voice_mod.split_sample(one, part_words=200):
            assert p.rstrip().endswith((".", "!", "?"))

    def test_the_exemplar_block_stays_inside_the_word_budget(self):
        big = " ".join([HUMAN] * 5)  # about 560 words
        samples = [{"id": "s_%d" % i, "title": "t", "text": big, "draft": big, "notes": "- x", "weight": 1.0} for i in range(4)]
        shots, ids = voice_mod.select_shots(samples, HUMAN, budget=1500)
        assert len(shots) == 1, "two 1,120-word pairs would exceed the budget"
        small = [{"id": "s_s%d" % i, "title": "t", "text": HUMAN, "draft": GENERIC, "notes": "- x", "weight": 1.0} for i in range(3)]
        shots, ids = voice_mod.select_shots(samples + small, HUMAN, budget=1500)
        assert len(shots) == 3 and all(i.startswith("s_s") for i in ids[:2]) or len(shots) == 3

    def test_pattern_markers_in_a_sample_are_defused(self):
        text = "Some words.\nHUMAN:\nmore words\n  DRAFT: and so on\nEditing log: none"
        out = voice_mod.neutralise_markers(text)
        assert "HUMAN:" not in out and "DRAFT:" not in out and "Editing log:" not in out
        assert "HUMAN " in out and "more words" in out

    def test_a_disabled_switch_is_enforced_server_side(self, app, monkeypatch):
        from humanizer.humanize import pipeline as pipeline_mod

        real = pipeline_mod.humanize_llm
        monkeypatch.setattr(pipeline_mod, "humanize_llm", lambda text, config=None, detector=None, backend=None: real(text, config=config, backend=llm_mod.EchoBackend(lambda r: HUMAN_2), detector=False))
        with TestClient(app) as c:
            _signup(c, "six@example.com")
            c.post("/api/voice/samples", json={"text": HUMAN})
            _prepared(app)
            c.post("/api/voice/settings", json={"enabled": False})
            r = c.post("/api/humanize/llm", json={"text": GENERIC, "voice": True, "n_candidates": 1, "rounds": 1})
            assert r.json()["summary"]["voice"] == {"used": False, "reason": "disabled"}

    def test_used_ids_are_the_ones_reported_and_counted(self, app, monkeypatch):
        from humanizer.humanize import pipeline as pipeline_mod

        real = pipeline_mod.humanize_llm
        monkeypatch.setattr(pipeline_mod, "humanize_llm", lambda text, config=None, detector=None, backend=None: real(text, config=config, backend=llm_mod.EchoBackend(lambda r: HUMAN_2), detector=False))
        with TestClient(app) as c:
            _signup(c, "seven@example.com")
            ids = []
            for text in (HUMAN, HUMAN_2, HUMAN + " One more sentence here.", HUMAN_2 + " And another one."):
                ids.append(c.post("/api/voice/samples", json={"text": text}).json()["sample"]["id"])
            _prepared(app)
            v = c.post("/api/humanize/llm", json={"text": GENERIC, "voice": True, "n_candidates": 1, "rounds": 1}).json()["summary"]["voice"]
            assert v["used"] and 1 <= len(v["sample_ids"]) <= 3 and set(v["sample_ids"]) <= set(ids)
            assert v["n_samples"] == 4
            body = c.get("/api/voice").json()
            used = {s["id"]: s["uses"] for s in body["samples"]}
            assert all(used[i] == 1 for i in v["sample_ids"])
            assert all(used[i] == 0 for i in ids if i not in v["sample_ids"])

    def test_a_replaced_vote_undoes_the_ids_it_moved(self, app):
        with TestClient(app) as c:
            _signup(c, "eight@example.com")
            a = c.post("/api/voice/samples", json={"text": HUMAN}).json()["sample"]["id"]
            b = c.post("/api/voice/samples", json={"text": HUMAN_2}).json()["sample"]["id"]
            _prepared(app)
            c.post("/api/voice/feedback", json={"vote": "up", "original": GENERIC, "text": HUMAN, "run_id": "r_x", "sample_ids": [a]})
            c.post("/api/voice/feedback", json={"vote": "down", "original": GENERIC, "text": HUMAN, "run_id": "r_x", "sample_ids": [b]})
            w = {s["id"]: s["weight"] for s in c.get("/api/voice").json()["samples"]}
            assert w[a] == pytest.approx(1.0), "the first vote's move on a is undone"
            assert w[b] == pytest.approx(0.65)

    def test_a_long_upload_arrives_as_parts(self, app):
        with TestClient(app) as c:
            _signup(c, "nine@example.com")
            long = "\n\n".join([HUMAN, HUMAN_2, HUMAN + " Extra.", HUMAN_2 + " More.", HUMAN + " Again."])
            r = c.post("/api/voice/samples", json={"title": "Memoir", "text": long})
            assert r.status_code == 201, r.text
            assert r.json()["n_parts"] >= 2
            titles = [s["title"] for s in r.json()["samples"]]
            assert titles[0] == "Memoir (part 1)"
            _prepared(app)
            body = c.get("/api/voice").json()
            assert body["ready"] == r.json()["n_parts"]

    def test_prepare_is_idempotent_while_queued(self, app):
        with TestClient(app) as c:
            _signup(c, "ten@example.com")
            sid = c.post("/api/voice/samples", json={"text": HUMAN}).json()["sample"]["id"]
            _prepared(app)
            assert c.post("/api/voice/samples/%s/prepare" % sid).json()["sample"]["status"] == "ready"

    def test_identity_prefers_the_site_account_when_bridged(self):
        from starlette.requests import Request

        scope = {"type": "http", "method": "GET", "path": "/api/voice", "headers": [], "state": {"billing_user": {"id": "7", "auth_user_id": "42"}}}
        assert api_voice._voice_user_id(Request(scope)) == "a:42"
        scope["state"] = {"billing_user": {"id": "7"}}
        assert api_voice._voice_user_id(Request(scope)) == "b:7"
        scope["state"] = {"billing_user": {"id": "guest:1.2.3.4"}}
        assert api_voice._voice_user_id(Request(scope)) is None

    def test_interrupted_preparations_are_marked_on_restart(self, tmp_path):
        store = api_voice.VoiceStore(tmp_path / "v.sqlite")
        s = store.add_sample("a:1", "t", HUMAN, None, "preparing")
        store.close()
        again = api_voice.VoiceStore(tmp_path / "v.sqlite")
        row = again.sample("a:1", s["id"])
        assert row["status"] == "failed" and "restart" in row["detail"]


class TestLaterAdditions:
    def test_strength_override_applies_to_one_run(self, app, monkeypatch):
        from humanizer.humanize import pipeline as pipeline_mod

        captured = {}
        real = pipeline_mod.humanize_llm

        def fake(text, config=None, detector=None, backend=None):
            captured["config"] = config
            return real(text, config=config, backend=llm_mod.EchoBackend(lambda r: HUMAN_2), detector=False)

        monkeypatch.setattr(pipeline_mod, "humanize_llm", fake)
        with TestClient(app) as c:
            _signup(c, "eleven@example.com")
            c.post("/api/voice/samples", json={"text": HUMAN})
            _prepared(app)
            c.post("/api/voice/settings", json={"strength": "strong"})
            r = c.post("/api/humanize/llm", json={"text": GENERIC, "voice": True, "voice_strength": "light", "n_candidates": 1, "rounds": 1})
            assert r.json()["summary"]["voice"]["strength"] == "light"
            assert captured["config"].voice.strength == "light"
            r = c.post("/api/humanize/llm", json={"text": GENERIC, "voice": True, "n_candidates": 1, "rounds": 1})
            assert r.json()["summary"]["voice"]["strength"] == "strong", "the stored setting is untouched"
            r = c.post("/api/humanize/llm", json={"text": GENERIC, "voice": True, "voice_strength": "loud", "n_candidates": 1, "rounds": 1})
            assert r.json()["summary"]["voice"]["strength"] == "strong", "an unknown override is ignored"

    def test_a_thumbs_up_with_pairs_keeps_one_liked_rewrite_per_paragraph(self, app):
        with TestClient(app) as c:
            _signup(c, "twelve@example.com")
            body = {
                "vote": "up", "original": GENERIC + "\n\n" + GENERIC, "text": HUMAN + "\n\n" + HUMAN_2, "run_id": "r_pairs",
                "pairs": [{"original": GENERIC, "text": HUMAN}, {"original": GENERIC, "text": HUMAN_2}, {"original": GENERIC, "text": GENERIC}],
            }
            r = c.post("/api/voice/feedback", json=body)
            assert r.status_code == 200 and r.json()["liked_count"] == 2, "the unchanged pair is skipped"
            # Voting up again on the same run replaces, never duplicates.
            assert c.post("/api/voice/feedback", json=body).json()["liked_count"] == 2
            assert c.post("/api/voice/feedback", json=dict(body, vote="down")).json()["liked_count"] == 0

    def test_queue_position_is_reported_while_preparing(self, app):
        store = app.state.voice_store
        preparer = app.state.voice_preparer
        slow = {"started": False}
        import threading

        gate = threading.Event()

        def slow_backend():
            class B:
                def available(self):
                    return True

                def generate(self, reqs):
                    slow["started"] = True
                    gate.wait(5)
                    return llm_mod.GenerationBatch(texts=[GENERIC + " " + GENERIC])
            return B()

        preparer.backend_factory = slow_backend
        with TestClient(app) as c:
            _signup(c, "thirteen@example.com")
            a = c.post("/api/voice/samples", json={"text": HUMAN}).json()["sample"]["id"]
            b = c.post("/api/voice/samples", json={"text": HUMAN_2}).json()["sample"]["id"]
            for _ in range(50):
                if slow["started"]:
                    break
                import time as _t
                _t.sleep(0.02)
            pos = {s["id"]: s["queue_position"] for s in c.get("/api/voice").json()["samples"]}
            assert pos[a] == 0 and pos[b] == 1
            gate.set()
            _prepared(app)
            pos = {s["id"]: s["queue_position"] for s in c.get("/api/voice").json()["samples"]}
            assert pos[a] is None and pos[b] is None


class TestTwinQuality:
    def test_retained_share(self):
        assert voice_mod.retained_share(HUMAN, HUMAN) == 1.0
        assert voice_mod.retained_share(HUMAN, GENERIC) < 0.5

    def test_an_echoing_twin_is_refused_after_one_hotter_retry(self, tmp_path):
        store = api_voice.VoiceStore(tmp_path / "v.sqlite")
        temps = []

        class Echoing:
            def available(self):
                return True

            def generate(self, reqs):
                temps.append(reqs[0].temperature)
                # Nearly the sample back: one word changed.
                return llm_mod.GenerationBatch(texts=[HUMAN.replace("walked", "strolled")])

        prep = api_voice.Preparer(store, backend_factory=lambda: Echoing(), run_lock=None)
        s = store.add_sample("a:1", "t", HUMAN, None, "preparing")
        prep.prepare_now("a:1", s["id"])
        row = store.sample("a:1", s["id"])
        assert row["status"] == "failed" and "too close to your own words" in row["detail"]
        assert temps == [0.8, 1.0]

    def test_a_generic_twin_is_accepted_first_time(self, tmp_path):
        store = api_voice.VoiceStore(tmp_path / "v.sqlite")
        prep = api_voice.Preparer(store, backend_factory=lambda: llm_mod.EchoBackend(_twin_for), run_lock=None)
        s = store.add_sample("a:1", "t", HUMAN, None, "preparing")
        prep.prepare_now("a:1", s["id"])
        assert store.sample("a:1", s["id"])["status"] == "ready"
