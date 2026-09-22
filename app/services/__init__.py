"""
Service layer that creates and stores the movie catalog and recommendation
engine once per application for efficient and testable access.
"""

from __future__ import annotations

from flask import Flask, current_app

from app.services.catalog import Movie, MovieCatalog
from app.services.recommender import (
    ColdStartError,
    Recommendation,
    RecommendationEngine,
    SimilarityIndex,
    ContentSimilarityIndex,
    to_dicts,
)

__all__ = [
    "ColdStartError",
    "Movie",
    "MovieCatalog",
    "Recommendation",
    "RecommendationEngine",
    "SimilarityIndex",
    "ContentSimilarityIndex",
    "get_catalog",
    "get_engine",
    "init_services",
    "to_dicts",
]


def init_services(app: Flask) -> RecommendationEngine:
    """Build the catalog and engine once, at application start-up."""
    catalog = MovieCatalog(app.config["MOVIES_CSV"])
    index = SimilarityIndex(app.config["SIMILARITY_NPZ"], app.config["MODEL_META_JSON"])
    content_index = ContentSimilarityIndex(app.config["CONTENT_SIMILARITY_NPZ"])
    engine = RecommendationEngine(catalog, index, content_index)

    app.extensions["movie_catalog"] = catalog
    app.extensions["recommendation_engine"] = engine
    return engine


def get_catalog() -> MovieCatalog:
    return current_app.extensions["movie_catalog"]


def get_engine() -> RecommendationEngine:
    return current_app.extensions["recommendation_engine"]
