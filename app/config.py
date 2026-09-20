"""
Application configuration.

Every value can be overridden with an environment variable (see `.env.example`).
The defaults are chosen so that `python run.py` works on a fresh clone without
any configuration at all, while still failing loudly if the app is started in
production mode with the development secret key still in place.

Two rules make the `.env` file safe to edit by hand:

* A variable that is present but blank (for example `DATA_DIR=`) is treated
  exactly like a variable that is absent, so the built-in default applies.
* Relative paths (for example `DATA_DIR=data/processed`) are resolved against
  the project folder, never against whichever directory the shell happens to
  be in.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
DEV_SECRET_KEY = "dev-only-insecure-secret-key"

# `.env` is read here, at the top of the module that consumes it, rather than
# in `app/__init__.py`. The Config class below reads os.environ while the class
# body executes (i.e. at import time), so the file has to be loaded first no
# matter which module happens to import `app.config`. Variables already set in
# the real environment win over the file (override=False).
load_dotenv(BASE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    """Read a string setting; a missing *or blank* variable means 'use default'."""
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back on anything unparseable."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    """Read a folder setting; blank means default, relative means 'inside the project'."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


class Config:
    """Settings shared by every environment."""

    # --- Security -------------------------------------------------------
    SECRET_KEY = _env_str("SECRET_KEY", DEV_SECRET_KEY)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    # --- Database -------------------------------------------------------
    SQLALCHEMY_DATABASE_URI = _env_str(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'instance' / 'site.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Dataset --------------------------------------------------------
    # `data/processed` holds the artifacts produced by scripts/build_model.py.
    DATA_DIR = _env_path("DATA_DIR", BASE_DIR / "data" / "processed")
    MOVIES_CSV = DATA_DIR / "movies.csv"
    SIMILARITY_NPZ = DATA_DIR / "item_similarity.npz"
    MODEL_META_JSON = DATA_DIR / "model_meta.json"

    # --- Uploads --------------------------------------------------------
    PROFILE_PIC_FOLDER = BASE_DIR / "app" / "static" / "profile_pics"
    PROFILE_PIC_SIZE = (125, 125)
    MAX_CONTENT_LENGTH = 4 * 1024 * 1024  # reject uploads larger than 4 MB

    # --- Behaviour ------------------------------------------------------
    WATCHLIST_MAX_ITEMS = _env_int("WATCHLIST_MAX_ITEMS", 20)
    SEARCH_MIN_QUERY_LENGTH = _env_int("SEARCH_MIN_QUERY_LENGTH", 2)
    SEARCH_RESULT_LIMIT = _env_int("SEARCH_RESULT_LIMIT", 24)
    RECOMMENDER_RESULT_LIMIT = _env_int("RECOMMENDER_RESULT_LIMIT", 12)
    PERSONAL_RESULT_LIMIT = _env_int("PERSONAL_RESULT_LIMIT", 12)
    HOMEPAGE_FEATURED_COUNT = _env_int("HOMEPAGE_FEATURED_COUNT", 15)
    ADMIN_PAGE_SIZE = _env_int("ADMIN_PAGE_SIZE", 25)

    # --- Default admin account (created on first run if missing) --------
    ADMIN_USERNAME = _env_str("ADMIN_USERNAME", "admin")
    ADMIN_EMAIL = _env_str("ADMIN_EMAIL", "admin@moviehub.com")
    ADMIN_PASSWORD = _env_str("ADMIN_PASSWORD", "Admin123!")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True

    def __init__(self):
        if Config.SECRET_KEY == DEV_SECRET_KEY:
            raise RuntimeError(
                "Refusing to start in production with the development SECRET_KEY. "
                "Set a real SECRET_KEY environment variable first."
            )


class TestingConfig(Config):
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


CONFIG_BY_NAME = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config(name: str | None = None):
    """Resolve a config class by name, defaulting to FLASK_ENV then development."""
    name = name or _env_str("FLASK_ENV", "development")
    return CONFIG_BY_NAME.get(name, DevelopmentConfig)
