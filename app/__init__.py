"""
Creates and configures the Flask application in a controlled startup order,
supporting clean imports, isolated testing and flexible configurations.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, render_template

from app.config import get_config
from app.extensions import bcrypt, db, login_manager

__version__ = "2.0.0"


def create_app(config_name: str | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(get_config(config_name)())

    _configure_logging(app)
    _ensure_instance_folder(app)
    _register_extensions(app)
    _register_services(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_template_globals(app)

    from app import bootstrap

    bootstrap.init_app_data(app)
    return app


def _configure_logging(app: Flask) -> None:
    logging.basicConfig(
        level=logging.DEBUG if app.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Werkzeug logs every static asset at INFO; that drowns out our own output.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _ensure_instance_folder(app: Flask) -> None:
    """Create the folder holding a `sqlite:///...` database file, if needed."""
    uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    prefix = "sqlite:///"
    if not uri.startswith(prefix) or uri.endswith(":memory:"):
        return
    database_path = Path(uri[len(prefix):])
    if database_path.parent and str(database_path.parent) != ".":
        os.makedirs(database_path.parent, exist_ok=True)


def _register_extensions(app: Flask) -> None:
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)


def _register_services(app: Flask) -> None:
    from app.services import init_services

    init_services(app)


def _register_blueprints(app: Flask) -> None:
    from app.views import register_all

    register_all(app)


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_error):
        return render_template("errors/404.html", title="Not found"), 404

    @app.errorhandler(413)
    def payload_too_large(_error):
        return render_template("errors/413.html", title="File too large"), 413

    @app.errorhandler(500)
    def server_error(error):
        app.logger.exception("Unhandled error", exc_info=error)
        db.session.rollback()
        return render_template("errors/500.html", title="Server error"), 500


def _register_template_globals(app: Flask) -> None:
    """Expose a few helpers to every template without passing them per-route."""
    from flask import url_for

    from app.utils.youtube import trailer_search_url

    def poster(movie) -> str:
        """Return a movie's poster URL or a generated placeholder if unavailable."""
        
        if movie is None:
            return url_for("static", filename="default_movie.jpg")
        url = (movie.get("poster_url") if isinstance(movie, dict) else getattr(movie, "poster_url", "")) or ""
        if url:
            return url
        movie_id = movie.get("movie_id") if isinstance(movie, dict) else getattr(movie, "movie_id", "")
        return url_for("main.poster_placeholder", movie_id=movie_id)

    def star_fill(rating, position: int) -> str:
        """How full star number `position` (1-5) should be: full, half or empty."""
        try:
            value = float(rating or 0)
        except (TypeError, ValueError):
            return "empty"
        if value >= position:
            return "full"
        return "half" if value >= position - 0.5 else "empty"

    @app.context_processor
    def inject_globals():
        return {
            "app_version": __version__,
            "poster": poster,
            "star_fill": star_fill,
            "trailer_search_url": trailer_search_url,
        }
