"""The event streams send a comment frame during long silences so proxies
keep the connection open (2026-10-07)."""

import time

from humanizer.api import server


def _slow_events():
    yield "a"
    time.sleep(0.35)
    yield "b"
    time.sleep(0.05)
    yield "c"


def test_keepalive_fills_silence_and_keeps_order():
    out = list(server._with_keepalive(_slow_events(), interval=0.1))
    events = [x for x in out if x is not None]
    assert events == ["a", "b", "c"]
    # At least two keepalives landed inside the 350 ms gap, none after "c".
    gap = out[out.index("a") + 1 : out.index("b")]
    assert len(gap) >= 2 and all(x is None for x in gap)
    assert out[-1] == "c"


def test_keepalive_reraises_source_errors_in_order():
    def bad():
        yield "ok"
        raise RuntimeError("model gone")

    gen = server._with_keepalive(bad(), interval=0.1)
    assert next(gen) == "ok"
    try:
        next(gen)
    except RuntimeError as exc:
        assert "model gone" in str(exc)
    else:
        raise AssertionError("the source error was swallowed")


def test_closing_the_consumer_stops_the_source():
    seen = []

    def src():
        for i in range(1000):
            seen.append(i)
            time.sleep(0.01)
            yield i

    gen = server._with_keepalive(src(), interval=0.1)
    assert next(gen) == 0
    gen.close()
    n = len(seen)
    time.sleep(0.2)
    assert len(seen) <= n + 2  # the pump stopped at its next event


def test_keepalive_frame_is_an_sse_comment():
    assert server.SSE_KEEPALIVE.startswith(":") and server.SSE_KEEPALIVE.endswith("\n\n")
