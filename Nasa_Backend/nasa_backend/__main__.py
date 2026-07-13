# nasa_backend/__main__.py
"""python -m nasa_backend — serve the API (and, once built, the React UI).
Replaces the old `python3` entry point into the now-deleted Dash monolith.
debug/reloader intentionally off (CodeQL: Werkzeug debugger must never face
a network)."""
import os

from nasa_backend.api import create_app


def main():
    app = create_app()
    host = os.environ.get("NASA_HOST", "127.0.0.1")
    port = int(os.environ.get("NASA_PORT", "8050"))
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
