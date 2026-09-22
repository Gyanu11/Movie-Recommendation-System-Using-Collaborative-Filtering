"""
Hybrid item-based collaborative and content-based recommendation engine.
Precomputed movie similarities are used to recommend movies similar to a
selected movie or based on a user's ratings, while excluding already rated
or watchlisted movies.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from app.services.catalog import Movie, MovieCatalog

log = logging.getLogger(__name__)

# Ratings are on MovieLens' 0.5-5.0 scale. Centring on 3.0 means "liked it"
# contributes positively and "disliked it" contributes negatively.
RATING_MIDPOINT = 3.0
# Weight given to a watchlisted-but-unrated movie: a mild positive signal.
IMPLICIT_WATCHLIST_RATING = 4.0


@dataclass(slots=True)
class Recommendation:
    """A ranked result, ready for a template."""

    movie: Movie
    score: float
    because_of: str = ""  # title of the user's movie that contributed most

    def as_dict(self) -> dict:
        data = self.movie.as_dict()
        data["similarity_score"] = round(self.score * 100, 1)
        data["because_of"] = self.because_of
        return data


class SimilarityIndex:
    """Read-only access to the neighbour lists in `item_similarity.npz`."""

    def __init__(self, npz_path: str | Path, meta_path: str | Path | None = None):
        self.npz_path = Path(npz_path)
        self.meta_path = Path(meta_path) if meta_path else None
        self._lock = threading.RLock()
        self._row_of: dict[str, int] = {}
        self._neighbour_ids: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self._neighbour_scores: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.meta: dict = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.npz_path.exists():
                log.warning(
                    "%s not found. Recommendations are disabled until you run "
                    "'python scripts/build_model.py'.",
                    self.npz_path,
                )
                return

            with np.load(self.npz_path) as payload:
                movie_ids = payload["movie_ids"]
                self._neighbour_ids = payload["neighbour_ids"]
                self._neighbour_scores = payload["neighbour_scores"].astype(np.float32)

            self._row_of = {str(mid): row for row, mid in enumerate(movie_ids)}
            log.info(
                "Similarity model loaded: %s movies x top-%s neighbours",
                len(self._row_of),
                self._neighbour_ids.shape[1] if self._neighbour_ids.size else 0,
            )

            if self.meta_path and self.meta_path.exists():
                try:
                    self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    log.warning("Could not read model metadata at %s", self.meta_path)

    @property
    def is_ready(self) -> bool:
        return bool(self._row_of)

    @property
    def movie_count(self) -> int:
        return len(self._row_of)

    def has(self, movie_id) -> bool:
        return str(movie_id).strip() in self._row_of

    def neighbours(self, movie_id, limit: int | None = None) -> list[tuple[str, float]]:
        """Top neighbours of one movie as (movie_id, cosine similarity) pairs."""
        row = self._row_of.get(str(movie_id).strip())
        if row is None:
            return []
        ids = self._neighbour_ids[row]
        scores = self._neighbour_scores[row]
        if limit is not None:
            ids, scores = ids[:limit], scores[:limit]
        return [(str(mid), float(score)) for mid, score in zip(ids, scores) if score > 0]


class ContentSimilarityIndex:
    """Read-only access to TF-IDF genre and keyword neighbours."""

    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        self._row_of: dict[str, int] = {}
        self._neighbour_ids: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self._neighbour_scores: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.npz_path.exists():
            log.warning("%s not found. Content recommendations are disabled until you run 'python scripts/build_model.py'.", self.npz_path)
            return
        with np.load(self.npz_path) as payload:
            movie_ids = payload["movie_ids"]
            self._neighbour_ids = payload["neighbour_ids"]
            self._neighbour_scores = payload["neighbour_scores"].astype(np.float32)
        self._row_of = {str(mid): row for row, mid in enumerate(movie_ids)}
        log.info("Content similarity model loaded: %s movies", len(self._row_of))

    @property
    def is_ready(self) -> bool:
        return bool(self._row_of)

    def neighbours(self, movie_id, limit: int | None = None) -> list[tuple[str, float]]:
        row = self._row_of.get(str(movie_id).strip())
        if row is None:
            return []
        ids = self._neighbour_ids[row]
        scores = self._neighbour_scores[row]
        if limit is not None:
            ids, scores = ids[:limit], scores[:limit]
        return [(str(mid), float(score)) for mid, score in zip(ids, scores) if score > 0]


class RecommendationEngine:
    """Turns the similarity index plus a user profile into ranked movies."""

    def __init__(self, catalog: MovieCatalog, index: SimilarityIndex, content_index: ContentSimilarityIndex | None = None):
        self.catalog = catalog
        self.index = index
        self.content_index = content_index

    def _neighbours(self, movie_id, limit: int) -> dict[str, float]:
        """Blend collaborative and content similarity, preserving both signals."""
        scores: dict[str, float] = {}
        for neighbour_id, score in self.index.neighbours(movie_id, limit=limit * 2):
            scores[neighbour_id] = scores.get(neighbour_id, 0.0) + 0.7 * score
        if self.content_index and self.content_index.is_ready:
            for neighbour_id, score in self.content_index.neighbours(movie_id, limit=limit * 2):
                scores[neighbour_id] = scores.get(neighbour_id, 0.0) + 0.3 * score
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    # "More like this"
    def similar_to(self, movie_id, limit: int = 12) -> list[Recommendation]:
        """Movies whose audiences overlap most with the given movie's."""
        neighbours = self._neighbours(movie_id, limit=limit)
        results: list[Recommendation] = []
        for neighbour_id, score in neighbours.items():
            movie = self.catalog.get(neighbour_id)
            if movie is not None:
                results.append(Recommendation(movie=movie, score=score))
            if len(results) == limit:
                break
        return results

    def recommend_by_title(self, title_query: str, limit: int = 12) -> tuple[Movie, list[Recommendation]]:
        """Resolve a typed title to a catalog movie, then recommend from it.

        Raises:
            LookupError: the title does not match anything in the catalog.
            ColdStartError: the title matched, but has no rating history to
                collaborate on (only possible for admin-added entries).
        """
        seed = self.catalog.find_best_match(title_query)
        if seed is None:
            raise LookupError(
                f"No movie matching '{title_query}' is in the catalog. "
                f"Try a different spelling, or search for it first."
            )

        results = self.similar_to(seed.movie_id, limit=limit)
        if not results:
            raise ColdStartError(seed)
        return seed, results

    # "Recommended for you"
    def for_user(
        self,
        ratings: Mapping[str, float],
        watchlist: Sequence[str] = (),
        limit: int = 12,
    ) -> list[Recommendation]:
        """Generate recommendations from a user's ratings and watchlist, or return an empty list 
        for an empty profile."""
        profile = self._build_profile(ratings, watchlist)
        if not profile:
            return []

        seen = set(profile)
        totals: dict[str, float] = defaultdict(float)
        best_source: dict[str, tuple[float, str]] = {}

        for source_id, weight in profile.items():
            source_title = ""
            for neighbour_id, similarity in self._neighbours(source_id, limit=50).items():
                if neighbour_id in seen:
                    continue
                contribution = similarity * weight
                totals[neighbour_id] += contribution

                # Remember which of the user's movies pulled this in hardest,
                # so the UI can explain the recommendation.
                if contribution > best_source.get(neighbour_id, (0.0, ""))[0]:
                    if not source_title:
                        source = self.catalog.get(source_id)
                        source_title = source.title if source else ""
                    best_source[neighbour_id] = (contribution, source_title)


       # Normalize scores by the user's total profile weight to keep them comparable
       # across users without changing the recommendation ranking.
        divisor = sum(abs(weight) for weight in profile.values()) or 1.0

        scored = [
            (movie_id, total / divisor) for movie_id, total in totals.items() if total > 0
        ]
        scored.sort(key=lambda item: item[1], reverse=True)

        results: list[Recommendation] = []
        for movie_id, score in scored:
            movie = self.catalog.get(movie_id)
            if movie is None:
                continue
            results.append(
                Recommendation(movie=movie, score=score, because_of=best_source.get(movie_id, (0.0, ""))[1])
            )
            if len(results) == limit:
                break
        return results

    def _build_profile(self, ratings: Mapping[str, float], watchlist: Sequence[str]) -> dict[str, float]:
        """Merge explicit ratings and implicit watchlist saves into weights."""
        profile: dict[str, float] = {}

        for movie_id, rating in ratings.items():
            key = str(movie_id).strip()
            if self.index.has(key):
                profile[key] = float(rating) - RATING_MIDPOINT

        for movie_id in watchlist:
            key = str(movie_id).strip()
            if key not in profile and self.index.has(key):
                profile[key] = IMPLICIT_WATCHLIST_RATING - RATING_MIDPOINT

        # A weight of exactly zero (a 3-star "it was fine") carries no signal.
        return {movie_id: weight for movie_id, weight in profile.items() if weight != 0}


class ColdStartError(Exception):
    """Raised when a movie exists in the catalog but has no rating history."""

    def __init__(self, movie: Movie):
        self.movie = movie
        super().__init__(
            f"'{movie.title}' has no rating history in the MovieLens data, so "
            f"collaborative recommendations are not available for it yet."
        )


def to_dicts(recommendations: Iterable[Recommendation]) -> list[dict]:
    """Convert recommendations for template consumption."""
    return [recommendation.as_dict() for recommendation in recommendations]
