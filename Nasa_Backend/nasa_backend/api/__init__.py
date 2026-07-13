# nasa_backend/api/__init__.py
"""Flask app factory: /api/* blueprint + the built React bundle at / (spec §9).
No import-time side effects — the model loads lazily on first /api/process."""
import os

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _webui_dir():
    env = os.environ.get("NASA_WEBUI_DIR")
    candidates = [env] if env else [
        os.path.abspath(os.path.join(_PKG_DIR, "..", "webui")),                      # Docker copy (PR ③)
        os.path.abspath(os.path.join(_PKG_DIR, "..", "..", "..", "nasa-frontend", "build")),  # dev checkout
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "index.html")):
            return c
    return None


def create_app():
    # static_folder=None: this package ships no static/ dir of its own, and
    # Flask's default /static/<path:filename> rule would otherwise be matched
    # ahead of the webui catch-all below, 404ing before it ever reaches the
    # built React bundle's own static/ assets.
    app = Flask(__name__, static_folder=None)
    # Dev-server origins only; in production the React bundle is same-origin
    # and never triggers CORS. Comma-separated env override.
    origins = [o.strip() for o in
               os.environ.get("NASA_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    CORS(app, resources={r"/api/*": {"origins": origins}}, supports_credentials=True)
    from nasa_backend.api.routes import api_bp
    app.register_blueprint(api_bp)

    webui = _webui_dir()
    if webui:
        @app.route("/")
        @app.route("/<path:path>")
        def webui_serve(path="index.html"):
            if path.startswith("api/"):
                return jsonify({"status": "error", "message": "Unknown API route"}), 404
            full = os.path.join(webui, path)
            if path != "index.html" and os.path.isfile(full):
                return send_from_directory(webui, path)
            return send_from_directory(webui, "index.html")
    else:
        @app.route("/")
        def webui_missing():
            return ("<h1>NASA Water Droplet backend</h1>"
                    "<p>UI not built — run `npm run build` in nasa-frontend/ "
                    "or set NASA_WEBUI_DIR. The /api/* endpoints are live.</p>"), 200
    return app
