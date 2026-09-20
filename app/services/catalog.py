"""
Movie catalog: the read side of the dataset.

`data/processed/movies.csv` is produced by `scripts/build_model.py` from the
raw MovieLens files. It is small (about 4,500 rows), so the whole thing is
parsed once at start-up into plain dictionaries and a handful of indexes:

    _by_id        movie_id -> Movie            O(1) detail lookups
    _titles       normalised title -> Movie    O(1) exact-title lookups
    _token_index  token -> set of movie ids    candidate generation for search

Every route reads through this object. Nothing in the request path touches the
filesystem unless an admin has just edited the catalog, in which case the
file's modification time changes and the cache reloads itself.

The runtime deliberately uses the standard library's `csv` module rather than
pandas: for a few thousand rows a DataFrame buys nothing, and dropping the
import cuts several seconds off start-up.
"""
from __future__ import annotations

import csv
import logging
import random
import threading
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from app.utils.text import normalise, tokenise

log = logging.getLogger(__name__)

CSV_COLUMNS = [
    "movie_id", "title", "year", "genres", "keywords",
    "rating_count", "rating_avg", "imdb_id", "tmdb_id", "poster_url",
]

# Scores used by `search()`, all on a single 0-1000 scale so that a threshold
# means the same thing whichever branch produced the score. Ordered bands keep
# an exact title above a prefix, a prefix above a substring, and so on.
_SCORE_EXACT_TITLE = 1000.0
_SCORE_TITLE_PREFIX = 800.0
_SCORE_TITLE_SUBSTRING = 600.0
_SCORE_TOKEN_BASE = 300.0     # plus up to 300 more for full token coverage
_SCORE_TOKEN_SPAN = 300.0
_SCORE_FUZZY_SPAN = 700.0     # a 0.72-1.0 ratio maps to roughly 504-700
_FUZZY_FLOOR = 0.72
_SPELLING_CUTOFF = 0.78        # how close a query word must be to a real title word

# `find_best_match` seeds the recommender, so it demands a confident hit: a
# substring match, near-total token coverage, or a close fuzzy match. Without
# this, a query like "not a real film" would latch onto any title containing
# the word "film" and recommend from it.
_MIN_CONFIDENT_SCORE = 500.0


@dataclass(slots=True)
class Movie:
    """One catalog entry. `slots` keeps 4,500 of these cheap to hold in RAM."""

    movie_id: str
    title: str
    year: str = ""
    genres: str = ""
    keywords: str = ""
    rating_count: int = 0
    rating_avg: float = 0.0
    imdb_id: str = ""
    tmdb_id: str = ""
    poster_url: str = ""
    normalised_title: str = field(default="", repr=False)

    @property
    def genre_list(self) -> list[str]:
        return [g.strip() for g in self.genres.split("|") if g.strip()]

    @property
    def keyword_list(self) -> list[str]:
        return [k.strip() for k in self.keywords.split("|") if k.strip()]

    @property
    def imdb_url(self) -> str:
        return f"https://www.imdb.com/title/{self.imdb_id}/" if self.imdb_id else ""

    def as_dict(self) -> dict:
        """Template-friendly view. Templates should never see the dataclass."""
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "genre_list": self.genre_list,
            "keywords": self.keywords,
            "keyword_list": self.keyword_list,
            "rating_count": self.rating_count,
            "rating_avg": self.rating_avg,
            "imdb_id": self.imdb_id,
            "imdb_url": self.imdb_url,
            "tmdb_id": self.tmdb_id,
            "poster_url": self.poster_url,
        }

    def to_row(self) -> dict:
        """Serialise back to the CSV shape used by `movies.csv`."""
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "keywords": self.keywords,
            "rating_count": self.rating_count,
            "rating_avg": f"{self.rating_avg:.3f}",
            "imdb_id": self.imdb_id,
            "tmdb_id": self.tmdb_id,
            "poster_url": self.poster_url,
        }


class MovieCatalog:
    """Thread-safe, self-refreshing, in-memory view of `movies.csv`."""

    def __init__(self, movies_csv: str | Path):
        self.movies_csv = Path(movies_csv)
        self._lock = threading.RLock()
        self._mtime: float | None = None
        self._movies: list[Movie] = []
        self._by_id: dict[str, Movie] = {}
        self._titles: dict[str, Movie] = {}
        self._token_index: dict[str, set[str]] = {}
        self._load(force=True)

    # ------------------------------------------------------------------ #
    # Loading and indexing
    # ------------------------------------------------------------------ #
    def _current_mtime(self) -> float | None:
        try:
            return self.movies_csv.stat().st_mtime
        except OSError:
            return None

    def _load(self, force: bool = False) -> None:
        with self._lock:
            mtime = self._current_mtime()
            if not force and mtime == self._mtime:
                return

            movies = list(self._read_rows())
            self._movies = movies
            self._by_id = {m.movie_id: m for m in movies}
            self._titles = {m.normalised_title: m for m in movies if m.normalised_title}

            token_index: dict[str, set[str]] = {}
            for movie in movies:
                for token in tokenise(movie.title):
                    token_index.setdefault(token, set()).add(movie.movie_id)
            self._token_index = token_index
            self._mtime = mtime
            log.info("Catalog loaded: %s movies from %s", len(movies), self.movies_csv)

    def _read_rows(self) -> Iterator[Movie]:
        if not self.movies_csv.exists():
            log.warning(
                "%s not found. Run 'python scripts/build_model.py' to generate it.", self.movies_csv
            )
            return

        with self.movies_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                movie_id = (row.get("movie_id") or "").strip()
                title = (row.get("title") or "").strip()
                if not movie_id or not title:
                    continue
                yield Movie(
                    movie_id=movie_id,
                    title=title,
                    year=(row.get("year") or "").strip(),
                    genres=(row.get("genres") or "").strip(),
                    keywords=(row.get("keywords") or "").strip(),
                    rating_count=_to_int(row.get("rating_count")),
                    rating_avg=_to_float(row.get("rating_avg")),
                    imdb_id=(row.get("imdb_id") or "").strip(),
                    tmdb_id=(row.get("tmdb_id") or "").strip(),
                    poster_url=(row.get("poster_url") or "").strip(),
                    normalised_title=normalise(title),
                )

    def reload(self) -> None:
        """Force a re-read. Called after any admin write."""
        self._load(force=True)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        self._load()
        return len(self._movies)

    def all_movies(self) -> list[Movie]:
        self._load()
        return list(self._movies)

    def get(self, movie_id) -> Movie | None:
        self._load()
        return self._by_id.get(str(movie_id).strip())

    def get_many(self, movie_ids: Iterable) -> list[Movie]:
        """Look up several ids at once, silently skipping any that are gone."""
        self._load()
        found = (self._by_id.get(str(mid).strip()) for mid in movie_ids)
        return [movie for movie in found if movie is not None]

    def exists(self, movie_id, exclude_id=None) -> bool:
        self._load()
        candidate = str(movie_id).strip()
        if exclude_id is not None and candidate == str(exclude_id).strip():
            return False
        return candidate in self._by_id

    def next_movie_id(self) -> str:
        self._load()
        numeric = [int(m.movie_id) for m in self._movies if m.movie_id.isdigit()]
        return str(max(numeric) + 1) if numeric else "1"

    def random_movies(self, count: int) -> list[Movie]:
        self._load()
        if not self._movies:
            return []
        return random.sample(self._movies, min(count, len(self._movies)))

    def random_movie(self) -> Movie | None:
        picked = self.random_movies(1)
        return picked[0] if picked else None

    def most_popular(self, limit: int, exclude: Sequence[str] = ()) -> list[Movie]:
        """Top titles by MovieLens rating volume.

        Used as the cold-start answer for a brand-new user: with no ratings and
        no watchlist there is nothing to collaborate on yet, so the honest
        fallback is "what most people watched".
        """
        self._load()
        skip = {str(mid).strip() for mid in exclude}
        ranked = sorted(self._movies, key=lambda m: m.rating_count, reverse=True)
        return [m for m in ranked if m.movie_id not in skip][:limit]

    def top_rated(self, limit: int, minimum_votes: int = 2000) -> list[Movie]:
        """Highest-rated titles that clear a vote threshold.

        A plain average would put obscure films with a handful of perfect
        scores on top, so a minimum vote count is required first.
        """
        self._load()
        eligible = [m for m in self._movies if m.rating_count >= minimum_votes]
        return sorted(eligible, key=lambda m: m.rating_avg, reverse=True)[:limit]

    def genres(self) -> list[str]:
        self._load()
        found: set[str] = set()
        for movie in self._movies:
            found.update(movie.genre_list)
        return sorted(found)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(self, query: str, limit: int = 24) -> list[Movie]:
        """Rank catalog titles against a free-text query."""
        return [movie for movie, _ in self.search_scored(query, limit)]

    def search_scored(self, query: str, limit: int = 24) -> list[tuple[Movie, float]]:
        """Rank catalog titles, returning each match with its score.

        The previous implementation ran `fuzzywuzzy.partial_ratio` five times
        per row over every row, for every query. This version narrows to a
        candidate set with an inverted token index first, and only reaches for
        fuzzy matching when the index misses - which is exactly where typo
        tolerance is needed and nowhere else.
        """
        self._load()
        needle = normalise(query)
        if not needle:
            return []

        tokens = tokenise(query)
        scores = self._score_candidates(needle, tokens, penalty=1.0)

        if not scores:
            # Nothing matched literally, so try correcting each query word
            # against the vocabulary of words that actually appear in titles.
            # Fixing "matrx" to "matrix" and reusing the index is far cheaper
            # and more accurate than comparing the query against all 4,500
            # titles character by character.
            corrected, confidence = self._correct_tokens(tokens)
            if corrected:
                scores = self._score_candidates(needle, corrected, penalty=confidence)

        if not scores:
            scores = self._fuzzy_title_scores(needle)

        # Popularity breaks ties without ever lifting a match into a higher
        # band, so the ordering of the bands above is preserved.
        ranked = sorted(
            scores.items(),
            key=lambda item: (item[1], self._by_id[item[0]].rating_count),
            reverse=True,
        )[:limit]
        return [(self._by_id[movie_id], score) for movie_id, score in ranked]

    def _score_candidates(self, needle: str, tokens: frozenset[str], penalty: float) -> dict[str, float]:
        """Score every movie reachable from `tokens` through the inverted index."""
        candidate_ids: set[str] = set()
        for token in tokens:
            candidate_ids |= self._token_index.get(token, set())

        scores: dict[str, float] = {}
        for movie_id in candidate_ids:
            movie = self._by_id[movie_id]
            title = movie.normalised_title
            if title == needle:
                score = _SCORE_EXACT_TITLE
            elif title.startswith(needle):
                score = _SCORE_TITLE_PREFIX
            elif needle in title:
                score = _SCORE_TITLE_SUBSTRING
            else:
                coverage = len(tokens & tokenise(movie.title)) / max(len(tokens), 1)
                score = _SCORE_TOKEN_BASE + _SCORE_TOKEN_SPAN * coverage
            scores[movie_id] = score * penalty

        exact = self._titles.get(needle)
        if exact is not None:
            scores[exact.movie_id] = _SCORE_EXACT_TITLE

        return scores

    def _correct_tokens(self, tokens: frozenset[str]) -> tuple[frozenset[str], float]:
        """Map misspelled query words onto real title words.

        Returns the corrected tokens and a 0-1 confidence, which discounts the
        resulting scores so a corrected match never outranks a literal one.
        """
        vocabulary = list(self._token_index)
        corrected: set[str] = set()
        confidences: list[float] = []

        for token in tokens:
            if token in self._token_index:
                corrected.add(token)
                confidences.append(1.0)
                continue
            close = get_close_matches(token, vocabulary, n=1, cutoff=_SPELLING_CUTOFF)
            if close:
                corrected.add(close[0])
                confidences.append(SequenceMatcher(None, token, close[0]).ratio())

        if not corrected:
            return frozenset(), 0.0
        return frozenset(corrected), sum(confidences) / len(confidences)

    def _fuzzy_title_scores(self, needle: str) -> dict[str, float]:
        """Whole-title similarity: the last resort when everything else misses."""
        matcher = SequenceMatcher()
        matcher.set_seq2(needle)
        scores: dict[str, float] = {}
        for movie in self._movies:
            matcher.set_seq1(movie.normalised_title)
            # quick_ratio is a cheap upper bound on ratio, so anything below the
            # floor is ruled out without running the expensive comparison.
            if matcher.quick_ratio() < _FUZZY_FLOOR:
                continue
            ratio = matcher.ratio()
            if ratio >= _FUZZY_FLOOR:
                scores[movie.movie_id] = _SCORE_FUZZY_SPAN * ratio
        return scores


    def find_best_match(self, query: str) -> Movie | None:
        """Single confident title match, used to resolve the recommender's seed.

        Returns None rather than a weak guess: recommending from the wrong film
        is worse than telling the user the title was not found.
        """
        results = self.search_scored(query, limit=1)
        if not results:
            return None
        movie, score = results[0]
        return movie if score >= _MIN_CONFIDENT_SCORE else None

    def suggest(self, prefix: str, limit: int = 8) -> list[Movie]:
        """Autocomplete: popular titles whose words start with `prefix`."""
        self._load()
        needle = normalise(prefix)
        if len(needle) < 2:
            return []
        matches = [
            movie
            for movie in self._movies
            if movie.normalised_title.startswith(needle) or f" {needle}" in movie.normalised_title
        ]
        matches.sort(key=lambda m: m.rating_count, reverse=True)
        return matches[:limit]

    # ------------------------------------------------------------------ #
    # Writes (admin CRUD)
    # ------------------------------------------------------------------ #
    def create(self, movie: Movie) -> Movie:
        with self._lock:
            if self.exists(movie.movie_id):
                raise ValueError(f"Movie ID '{movie.movie_id}' already exists.")
            movie.normalised_title = normalise(movie.title)
            self._write_all(self._movies + [movie])
            return movie

    def update(self, original_movie_id, movie: Movie) -> Movie:
        with self._lock:
            original = str(original_movie_id).strip()
            current = self.get(original)
            if current is None:
                raise ValueError(f"Movie '{original}' not found.")
            if movie.movie_id != original and self.exists(movie.movie_id):
                raise ValueError(f"Movie ID '{movie.movie_id}' already exists.")

            # Rating statistics come from MovieLens and are not editable, so
            # they are carried over rather than taken from the form.
            movie.rating_count = current.rating_count
            movie.rating_avg = current.rating_avg
            movie.normalised_title = normalise(movie.title)

            rows = [movie if m.movie_id == original else m for m in self._movies]
            self._write_all(rows)
            return movie

    def delete(self, movie_id) -> bool:
        with self._lock:
            target = str(movie_id).strip()
            remaining = [m for m in self._movies if m.movie_id != target]
            if len(remaining) == len(self._movies):
                return False
            self._write_all(remaining)
            return True

    def _write_all(self, movies: list[Movie]) -> None:
        """Rewrite `movies.csv` atomically, then refresh the in-memory copy.

        Writing to a temporary file and replacing it means a crash mid-write
        can never leave a half-written catalog on disk.
        """
        self.movies_csv.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.movies_csv.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for movie in movies:
                writer.writerow(movie.to_row())
        temporary.replace(self.movies_csv)
        self._load(force=True)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
