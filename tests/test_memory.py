"""Bounded model caches and the resident-model API.

Nothing here loads weights or imports torch or MLX. The caches are exercised
through their insertion helpers with dummy objects, the LLM registry through
`MlxBackend` with `load`/`unload` stubbed out, and the endpoints through
FastAPI's TestClient exactly as `tests/test_api.py` does.
"""

from __future__ import annotations

import sys

import pytest

from humanizer import memory
from humanizer.detectors import local
from humanizer.humanize import guided, llm, pretrained


@pytest.fixture(autouse=True)
def _clean_caches():
    local.clear_model_cache()
    pretrained.clear_model_cache()
    llm.release_backends()
    guided.clear_guide_cache()
    yield
    local.clear_model_cache()
    pretrained.clear_model_cache()
    llm.release_backends()
    guided.clear_guide_cache()


# --------------------------------------------------------------- memory module


class TestMaxResident:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("X_CAP", raising=False)
        assert memory.max_resident("X_CAP", 3) == 3

    @pytest.mark.parametrize("raw,expected", [("2", 2), ("0", 0), ("-4", 0), ("junk", 5), ("  7 ", 7), ("", 5)])
    def test_parses_env(self, monkeypatch, raw, expected):
        monkeypatch.setenv("X_CAP", raw)
        assert memory.max_resident("X_CAP", 5) == expected

    def test_release_memory_without_torch_or_mlx(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
        memory.release_memory(object(), [1, 2, 3])  # must not raise

    def test_rss_mb_is_positive_int(self):
        rss = memory.rss_mb()
        assert isinstance(rss, int) and rss > 0


# ------------------------------------------------------------ detectors.local


class TestDetectorLRU:
    def test_evicts_oldest_past_cap(self, monkeypatch):
        monkeypatch.setenv(local.MAX_RESIDENT_ENV, "2")
        local._remember(("modern", "a"), ("tok", "a"))
        local._remember(("modern", "b"), ("tok", "b"))
        assert local.loaded_models() == ["modern:a", "modern:b"]
        local._remember(("causal", "c"), ("tok", "c"))
        assert local.loaded_models() == ["causal:c", "modern:b"]

    def test_hit_refreshes_recency(self, monkeypatch):
        monkeypatch.setenv(local.MAX_RESIDENT_ENV, "2")
        local._remember(("modern", "a"), ("tok", "a"))
        local._remember(("modern", "b"), ("tok", "b"))
        assert local._recall(("modern", "a")) == ("tok", "a")
        local._remember(("modern", "c"), ("tok", "c"))
        # b was least recently used once a was recalled.
        assert local.loaded_models() == ["modern:a", "modern:c"]

    def test_zero_means_unlimited(self, monkeypatch):
        monkeypatch.setenv(local.MAX_RESIDENT_ENV, "0")
        for i in range(10):
            local._remember(("modern", f"m{i}"), ("tok", i))
        assert len(local.loaded_models()) == 10

    def test_default_cap_is_three(self, monkeypatch):
        monkeypatch.delenv(local.MAX_RESIDENT_ENV, raising=False)
        for i in range(5):
            local._remember(("modern", f"m{i}"), ("tok", i))
        assert local.loaded_models() == ["modern:m2", "modern:m3", "modern:m4"]

    def test_eviction_releases(self, monkeypatch):
        monkeypatch.setenv(local.MAX_RESIDENT_ENV, "1")
        released = []
        monkeypatch.setattr(local, "_release", lambda *objs: released.extend(objs))
        local._remember(("modern", "a"), "A")
        local._remember(("modern", "b"), "B")
        assert released == ["A"]
        local.clear_model_cache()
        assert released == ["A", "B"]
        assert local.loaded_models() == []

    def test_remember_returns_value(self):
        assert local._remember(("modern", "a"), ("tok", "a")) == ("tok", "a")


# ------------------------------------------------------- humanize.pretrained


class TestPretrainedEviction:
    def test_default_keeps_one(self, monkeypatch):
        monkeypatch.delenv(pretrained.MAX_RESIDENT_ENV, raising=False)
        pretrained._remember(("m1", "cpu", "float16"), ("tok", "m1"))
        pretrained._remember(("m2", "cpu", "float16"), ("tok", "m2"))
        assert pretrained.loaded_models() == ["m2"]

    def test_cap_from_env(self, monkeypatch):
        monkeypatch.setenv(pretrained.MAX_RESIDENT_ENV, "2")
        for name in ("m1", "m2", "m3"):
            pretrained._remember((name, "cpu", "float16"), ("tok", name))
        assert pretrained.loaded_models() == ["m2", "m3"]

    def test_eviction_releases(self, monkeypatch):
        monkeypatch.setenv(pretrained.MAX_RESIDENT_ENV, "1")
        released = []
        monkeypatch.setattr(pretrained, "release_memory", lambda *objs: released.extend(objs))
        pretrained._remember(("m1", "cpu", "float16"), "A")
        pretrained._remember(("m2", "cpu", "float16"), "B")
        assert released == ["A"]
        pretrained.clear_model_cache()
        assert released == ["A", "B"]


# --------------------------------------------------------------- humanize.llm


@pytest.fixture
def stub_mlx(monkeypatch):
    """`MlxBackend` that never touches mlx: load/unload just flip a flag."""
    calls = {"load": [], "unload": []}

    def fake_load(self):
        self._model = object()
        self._tokenizer = object()
        calls["load"].append(self.model_name)

    def fake_unload(self):
        self._model = None
        self._tokenizer = None
        self._batch_fn = None
        calls["unload"].append(self.model_name)

    monkeypatch.setattr(llm.MlxBackend, "load", fake_load)
    monkeypatch.setattr(llm.MlxBackend, "unload", fake_unload)
    return calls


class TestBackendRegistry:
    def test_same_args_same_object(self, stub_mlx, monkeypatch):
        monkeypatch.setenv(llm.MAX_RESIDENT_ENV, "1")
        a = llm.backend_for("faithful")
        b = llm.backend_for("faithful")
        assert a is b
        assert llm.backend_for("faithful", llm.DEFAULT_MODEL) is a

    def test_different_style_unloads_previous(self, stub_mlx, monkeypatch):
        monkeypatch.setenv(llm.MAX_RESIDENT_ENV, "1")
        a = llm.backend_for("faithful")
        a.load()
        assert llm.loaded_backends() == [f"faithful:{llm.DEFAULT_MODEL}"]
        b = llm.backend_for("freeform")
        assert b is not a
        assert stub_mlx["unload"] == [llm.DEFAULT_MODEL]
        assert not a.loaded
        assert llm.loaded_backends() == []  # b is registered but not loaded

    def test_cap_two_keeps_both(self, stub_mlx, monkeypatch):
        monkeypatch.setenv(llm.MAX_RESIDENT_ENV, "2")
        a = llm.backend_for("faithful")
        b = llm.backend_for("freeform")
        a.load()
        b.load()
        assert stub_mlx["unload"] == []
        assert len(llm.loaded_backends()) == 2
        assert llm.backend_for("faithful") is a

    def test_explicit_model_gets_own_slot(self, stub_mlx, monkeypatch):
        monkeypatch.setenv(llm.MAX_RESIDENT_ENV, "0")
        a = llm.backend_for("faithful")
        b = llm.backend_for("faithful", "some/other-model")
        assert a is not b
        assert b.model_name == "some/other-model"

    def test_release_backends_unloads_all(self, stub_mlx, monkeypatch):
        monkeypatch.setenv(llm.MAX_RESIDENT_ENV, "0")
        llm.backend_for("faithful").load()
        llm.backend_for("freeform").load()
        llm.release_backends()
        assert sorted(stub_mlx["unload"]) == sorted([llm.DEFAULT_MODEL, llm.FREEFORM_MODEL])
        assert llm.loaded_backends() == []

    def test_default_backend_is_not_memoised(self, stub_mlx):
        assert llm.default_backend("faithful") is not llm.default_backend("faithful")


class TestMlxBackendUnload:
    def test_unload_clears_state_and_releases(self, monkeypatch):
        released = []
        monkeypatch.setattr(llm, "release_memory", lambda *objs: released.extend(objs))
        backend = llm.MlxBackend(model_name="x")
        backend._model, backend._tokenizer, backend._batch_fn = "M", "T", "B"
        assert backend.loaded
        backend.unload()
        assert not backend.loaded
        assert backend._batch_fn is None
        assert released == ["M", "T", "B"]

    def test_unload_when_never_loaded_is_a_noop(self):
        llm.MlxBackend(model_name="x").unload()


# ------------------------------------------------------------ humanize.guided


def test_clear_guide_cache():
    guided._GUIDE_CACHE[("guide", "m")] = object()
    guided.clear_guide_cache()
    assert guided._GUIDE_CACHE == {}


# ------------------------------------------------------------------------- API

pytest.importorskip("httpx", reason="fastapi's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from humanizer.api.server import create_app  # noqa: E402
from tests.test_api import REFERENCE_DIR, REPO_ROOT  # noqa: E402

MODEL_KEYS = {"detectors", "pretrained", "llm", "rss_mb"}


@pytest.fixture(scope="module")
def client():
    app = create_app(reference_dir=REFERENCE_DIR, web_dir=REPO_ROOT / "does-not-exist-web", auth=False)
    with TestClient(app) as c:
        yield c


class TestModelsApi:
    def test_get_models_shape(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == MODEL_KEYS
        assert isinstance(body["detectors"], list)
        assert isinstance(body["pretrained"], list)
        assert isinstance(body["llm"], list)
        assert isinstance(body["rss_mb"], int) and body["rss_mb"] > 0

    def test_get_models_reflects_caches(self, client, monkeypatch):
        monkeypatch.setenv(local.MAX_RESIDENT_ENV, "0")
        local._remember(("modern", "dummy"), ("tok", "m"))
        pretrained._remember(("para", "cpu", "float16"), ("tok", "m"))
        body = client.get("/api/models").json()
        assert body["detectors"] == ["modern:dummy"]
        assert body["pretrained"] == ["para"]

    def test_release_empties_everything(self, client, monkeypatch):
        monkeypatch.setenv(local.MAX_RESIDENT_ENV, "0")
        local._remember(("modern", "dummy"), ("tok", "m"))
        pretrained._remember(("para", "cpu", "float16"), ("tok", "m"))
        guided._GUIDE_CACHE[("guide", "g")] = object()
        r = client.post("/api/models/release")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == MODEL_KEYS | {"released"}
        assert body["released"] is True
        assert body["detectors"] == [] and body["pretrained"] == [] and body["llm"] == []
        assert guided._GUIDE_CACHE == {}

    def test_health_shape_unchanged(self, client):
        assert set(client.get("/api/health").json()) == {
            "status", "version", "syntax_backend", "references", "yardstick",
        }
