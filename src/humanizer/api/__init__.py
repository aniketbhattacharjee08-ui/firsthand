"""HTTP API for the Stage 0 analysis toolkit.

Thin transport layer over `humanizer.eval`, `humanizer.features`,
`humanizer.reference` and `humanizer.detectors`. It computes nothing the
library does not already compute; it only routes, validates and serialises.

    from humanizer.api import create_app
    app = create_app()

Run it with `humanizer serve --port 8000`.
"""

from .server import create_app, run

__all__ = ["create_app", "run"]
