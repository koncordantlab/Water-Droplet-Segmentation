# nasa_backend/api/__init__.py
"""Flask app factory. No import-time side effects — the model loads lazily on
the first /api/process (spec §6 seam #2)."""
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    # CORS for /api/* only, verbatim from the monolith; narrowed in the
    # CodeQL task later in this PR.
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    from nasa_backend.api.routes import api_bp
    app.register_blueprint(api_bp)
    return app
