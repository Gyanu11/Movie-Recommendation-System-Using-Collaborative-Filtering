"""
Service layer.

The catalog and the recommendation engine are expensive to build (a CSV parse
and a NumPy load) but completely read-only afterwards, so exactly one of each
is created per application and stored on `app.extensions`. Views reach them
through `get_catalog()` / `get_engine()` rather than importing a module-level
global, which keeps the app testable: each test builds its own instance.
"""
from __future__ import annotations

from flask import Flask, current_app

from app.services.catalog import Movie, MovieCatalog
from app.services.recommender import (
    ColdStartError,
    Recommendation,
    RecommendationEngine,
    SimilarityIndex,
    to_dicts,
)

__all__ = [
    "ColdStartError",
    "Movie",
    "MovieCatalog",
    "Recommendation",
    "RecommendationEngine",
    "SimilarityIndex",
    "get_catalog",
    "get_engine",
    "init_services",
    "to_dicts",
]


def init_services(app: Flask) -> RecommendationEngine:
    """Build the catalog and engine once, at application start-up."""
    catalog = MovieCatalog(app.config["MOVIES_CSV"])
    index = SimilarityIndex(app.config["SIMILARITY_NPZ"], app.config["MODEL_META_JSON"])
    engine = RecommendationEngine(catalog, index)

    app.extensions["movie_catalog"] = catalog
    app.extensions["recommendation_engine"] = engine
    return engine


def get_catalog() -> MovieCatalog:
    return current_app.extensions["movie_catalog"]


def get_engine() -> RecommendationEngine:
    return current_app.extensions["recommendation_engine"]
