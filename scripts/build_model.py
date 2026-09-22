"""
Offline model builder: turn the raw MovieLens 20M dataset into the two compact
artifacts the web app loads at start-up.

    data/raw/*.csv                      (~900 MB, not shipped with the project)
                |
                |  python scripts/build_model.py
                v
    data/processed/movies.csv           catalog metadata  (~1 MB)
    data/processed/item_similarity.npz  top-K neighbours  (~2 MB)
    data/processed/model_meta.json      how the model was built

Why a separate offline step?
----------------------------
The raw ratings file holds 20 million rows. Parsing it, building the
user-item matrix, and computing a 4,000 x 4,000 similarity matrix takes tens
of seconds and over a gigabyte of RAM. A web request cannot afford either, so
the expensive work happens once, here, and the Flask app only ever loads the
small pre-computed result. This is the standard "train offline, serve online"
split used by real recommender systems.

Algorithm
---------
Item-based collaborative filtering with adjusted cosine similarity.

1. Keep movies with at least MIN_RATINGS_PER_MOVIE ratings. Rarely-rated
   movies produce noisy, meaningless similarities.
2. Build a sparse item x user matrix R where R[i, u] is the rating user u gave
   movie i.
3. Subtract each movie's mean rating from its observed entries. This is the
   "adjusted" part: it removes the bias of universally-loved or universally-
   panned films, so what is left is how much *more or less* than usual each
   user liked that film.
4. L2-normalise every row, then compute S = R_norm @ R_norm.T. Because the
   rows are unit vectors, that product *is* the cosine similarity between
   every pair of movies.
5. Keep only the top-K neighbours of each movie. A dense 4,489 x 4,489 matrix
   is 80 MB; the top-50 neighbours of each movie are under 2 MB and give
   identical results for the top-N recommendations the app actually serves.

Usage
-----
    python scripts/build_model.py
    python scripts/build_model.py --min-ratings 1000 --neighbours 40
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Defaults chosen to balance catalog size against similarity quality:
# 500+ ratings keeps ~4,500 well-known films out of MovieLens' 27,000.
DEFAULT_MIN_RATINGS = 500
DEFAULT_NEIGHBOURS = 50
DEFAULT_CONTENT_NEIGHBOURS = 50
DEFAULT_KEYWORDS = 12
GENOME_RELEVANCE_FLOOR = 0.5
RATING_CHUNK_ROWS = 4_000_000  # keeps peak memory near 400 MB

TITLE_YEAR_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")

log = logging.getLogger("build_model")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(
            f"Missing required file: {path}\n"
            f"Download the MovieLens 20M dataset and place its CSV files in {RAW_DIR}/.\n"
            f"See data/raw/README.md for the exact list."
        )
    return path


def load_movies() -> pd.DataFrame:
    """movie.csv -> movie_id / raw_title / title / year / genres."""
    df = pd.read_csv(_require(RAW_DIR / "movie.csv"), dtype={"movieId": np.int32})
    df = df.rename(columns={"movieId": "movie_id", "title": "raw_title"})

    extracted = df["raw_title"].str.extract(TITLE_YEAR_RE)
    df["title"] = extracted["title"].fillna(df["raw_title"]).str.strip()
    df["year"] = pd.to_numeric(extracted["year"], errors="coerce").astype("Int64")

    # MovieLens stores titles with the article moved to the end ("Matrix, The").
    # Restoring natural word order makes search and display far more usable.
    df["title"] = df["title"].map(_restore_leading_article)
    df["genres"] = df["genres"].fillna("").replace("(no genres listed)", "")
    return df[["movie_id", "title", "year", "genres"]]


def _restore_leading_article(title: str) -> str:
    """'Matrix, The' -> 'The Matrix'; 'Godfather, The' -> 'The Godfather'."""
    match = re.match(r"^(?P<body>.+),\s*(?P<article>The|A|An|Le|La|Les|Il|El|L')$", title)
    if not match:
        return title
    article = match.group("article")
    separator = "" if article.endswith("'") else " "
    return f"{article}{separator}{match.group('body')}"


def aggregate_ratings(path: Path) -> pd.DataFrame:
    """Stream rating.csv and return per-movie rating count and mean."""
    totals: pd.DataFrame | None = None
    rows = 0
    for chunk in pd.read_csv(
        path,
        usecols=["movieId", "rating"],
        dtype={"movieId": np.int32, "rating": np.float32},
        chunksize=RATING_CHUNK_ROWS,
    ):
        rows += len(chunk)
        partial = chunk.groupby("movieId")["rating"].agg(["count", "sum"])
        totals = partial if totals is None else totals.add(partial, fill_value=0)
        del chunk, partial
        gc.collect()

    if totals is None:
        raise SystemExit("rating.csv contained no rows.")

    log.info("Read %s ratings across %s movies", f"{rows:,}", f"{len(totals):,}")
    totals["rating_count"] = totals["count"].astype(np.int32)
    totals["rating_avg"] = (totals["sum"] / totals["count"]).astype(np.float32)
    totals.index.name = "movie_id"
    return totals[["rating_count", "rating_avg"]].reset_index()


def load_rating_triples(path: Path, keep_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stream rating.csv again, keeping only ratings for `keep_ids`.

    Returns (row_index, user_index, rating) arrays ready for a COO matrix,
    plus the number of distinct users.
    """
    keep_index = pd.Index(keep_ids)
    row_of_movie = pd.Series(np.arange(len(keep_ids), dtype=np.int32), index=keep_ids)

    rows, users, values = [], [], []
    for chunk in pd.read_csv(
        path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
        chunksize=RATING_CHUNK_ROWS,
    ):
        chunk = chunk[chunk["movieId"].isin(keep_index)]
        rows.append(row_of_movie.reindex(chunk["movieId"]).to_numpy(dtype=np.int32))
        users.append(chunk["userId"].to_numpy(dtype=np.int32))
        values.append(chunk["rating"].to_numpy(dtype=np.float32))
        del chunk
        gc.collect()

    row_index = np.concatenate(rows)
    user_ids = np.concatenate(users)
    ratings = np.concatenate(values)

    # userIds are 1-based and dense in MovieLens, but re-indexing costs nothing
    # and protects against gaps in a subset of the data.
    unique_users, user_index = np.unique(user_ids, return_inverse=True)
    return row_index, user_index.astype(np.int32), ratings, len(unique_users)


def load_keywords(path: Path, keep_ids: np.ndarray, top_n: int) -> pd.Series:
    """genome_scores.csv -> the `top_n` most relevant tags per movie.

    MovieLens has no plot summary, cast, or director. The tag genome is the
    dataset's own descriptive layer: a relevance score from 0 to 1 for each of
    1,128 curated tags. The highest-scoring tags read as a usable description
    ("pixar animation | toys | kids and family") and are what the movie detail
    page shows in place of an overview.
    """
    if not path.exists():
        log.warning("genome_scores.csv not found - movies will have no keywords")
        return pd.Series(dtype=str, name="keywords")

    tags = pd.read_csv(_require(RAW_DIR / "genome_tags.csv")).set_index("tagId")["tag"]
    keep_index = pd.Index(keep_ids)

    kept = []
    for chunk in pd.read_csv(
        path,
        dtype={"movieId": np.int32, "tagId": np.int16, "relevance": np.float32},
        chunksize=5_000_000,
    ):
        kept.append(chunk[chunk["movieId"].isin(keep_index) & (chunk["relevance"] >= GENOME_RELEVANCE_FLOOR)])
        del chunk
        gc.collect()

    scores = pd.concat(kept, ignore_index=True)
    del kept
    gc.collect()

    scores = scores.sort_values(["movieId", "relevance"], ascending=[True, False])
    best = scores.groupby("movieId").head(top_n).copy()
    best["tag"] = best["tagId"].map(tags)
    keywords = best.groupby("movieId")["tag"].apply(lambda s: " | ".join(s)).rename("keywords")
    keywords.index.name = "movie_id"
    return keywords


def load_links(path: Path) -> pd.DataFrame:
    """link.csv -> IMDb and TMDB ids, used for outbound links and posters."""
    if not path.exists():
        return pd.DataFrame(columns=["movie_id", "imdb_id", "tmdb_id"])
    df = pd.read_csv(path, dtype={"movieId": np.int32})
    df = df.rename(columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"})
    df["imdb_id"] = df["imdb_id"].apply(lambda v: f"tt{int(v):07d}" if pd.notna(v) else "")
    df["tmdb_id"] = df["tmdb_id"].apply(lambda v: str(int(v)) if pd.notna(v) else "")
    return df[["movie_id", "imdb_id", "tmdb_id"]]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def build_similarity(
    row_index: np.ndarray,
    user_index: np.ndarray,
    ratings: np.ndarray,
    n_movies: int,
    n_users: int,
    neighbours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjusted-cosine item-item similarity, reduced to top-K neighbours.

    Returns (neighbour_rows, neighbour_scores), both shaped (n_movies, K).
    `neighbour_rows` holds *row positions*, which the caller maps to movie ids.
    """
    matrix = sp.csr_matrix(
        (ratings, (row_index, user_index)), shape=(n_movies, n_users), dtype=np.float32
    )
    matrix.sum_duplicates()
    log.info("Rating matrix: %s movies x %s users, %s non-zero", f"{n_movies:,}", f"{n_users:,}", f"{matrix.nnz:,}")

    per_row = np.diff(matrix.indptr)
    row_means = np.asarray(matrix.sum(axis=1)).ravel() / np.maximum(per_row, 1)

    # Centre each movie's ratings on its own mean, then L2-normalise the row.
    # Operating on `.data` directly keeps this O(nnz) instead of densifying.
    matrix.data -= np.repeat(row_means, per_row).astype(np.float32)
    norms = np.sqrt(np.add.reduceat(matrix.data**2, matrix.indptr[:-1]))
    norms[per_row == 0] = 1.0
    norms[norms == 0] = 1.0
    matrix.data /= np.repeat(norms, per_row).astype(np.float32)

    started = time.perf_counter()
    similarity = (matrix @ matrix.T).toarray()
    log.info("Cosine similarity computed in %.1fs", time.perf_counter() - started)
    del matrix
    gc.collect()

    np.fill_diagonal(similarity, -np.inf)  # never recommend the seed movie itself
    k = min(neighbours, n_movies - 1)

    # argpartition finds the top-k without fully sorting all 4,489 columns.
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(similarity, top, axis=1)
    order = np.argsort(-top_scores, axis=1)

    neighbour_rows = np.take_along_axis(top, order, axis=1).astype(np.int32)
    neighbour_scores = np.take_along_axis(top_scores, order, axis=1).astype(np.float32)
    return neighbour_rows, np.clip(neighbour_scores, 0.0, 1.0)


def build_content_similarity(catalog: pd.DataFrame, neighbours: int) -> tuple[np.ndarray, np.ndarray]:
    """Build TF-IDF cosine neighbours from each movie's genres and keywords."""
    documents = (catalog["genres"].fillna("") + " " + catalog["keywords"].fillna("")).str.lower()
    vocabulary: dict[str, int] = {}
    rows: list[dict[int, int]] = []
    for document in documents:
        counts: dict[int, int] = {}
        for token in re.findall(r"[a-z0-9']+", document):
            column = vocabulary.setdefault(token, len(vocabulary))
            counts[column] = counts.get(column, 0) + 1
        rows.append(counts)

    data, row_index, column_index = [], [], []
    document_frequency = np.zeros(len(vocabulary), dtype=np.float32)
    for row, counts in enumerate(rows):
        for column, count in counts.items():
            row_index.append(row)
            column_index.append(column)
            data.append(float(count))
            document_frequency[column] += 1
    matrix = sp.csr_matrix((data, (row_index, column_index)), shape=(len(rows), len(vocabulary)), dtype=np.float32)
    if matrix.shape[1]:
        matrix = matrix.multiply(np.log((1 + len(rows)) / (1 + document_frequency)) + 1)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    matrix = matrix.multiply((1.0 / norms)[:, None])
    similarity = (matrix @ matrix.T).toarray()
    np.fill_diagonal(similarity, -np.inf)
    k = min(neighbours, max(len(rows) - 1, 1))
    top = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    top_scores = np.take_along_axis(similarity, top, axis=1)
    order = np.argsort(-top_scores, axis=1)
    return (
        np.take_along_axis(top, order, axis=1).astype(np.int32),
        np.clip(np.take_along_axis(top_scores, order, axis=1), 0.0, 1.0).astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the MovieLens recommendation model.")
    parser.add_argument("--min-ratings", type=int, default=DEFAULT_MIN_RATINGS,
                        help=f"drop movies with fewer ratings (default: {DEFAULT_MIN_RATINGS})")
    parser.add_argument("--neighbours", type=int, default=DEFAULT_NEIGHBOURS,
                        help=f"neighbours stored per movie (default: {DEFAULT_NEIGHBOURS})")
    parser.add_argument("--content-neighbours", type=int, default=DEFAULT_CONTENT_NEIGHBOURS,
                        help=f"content neighbours stored per movie (default: {DEFAULT_CONTENT_NEIGHBOURS})")
    parser.add_argument("--keywords", type=int, default=DEFAULT_KEYWORDS,
                        help=f"genome tags kept per movie (default: {DEFAULT_KEYWORDS})")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    raw_dir, out_dir = args.raw_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    global RAW_DIR
    RAW_DIR = raw_dir

    started = time.perf_counter()

    log.info("Step 1/5  Reading movie metadata")
    movies = load_movies()

    log.info("Step 2/5  Aggregating ratings (streamed)")
    stats = aggregate_ratings(_require(raw_dir / "rating.csv"))
    popular = stats[stats["rating_count"] >= args.min_ratings]
    log.info("Kept %s of %s movies with >= %s ratings",
             f"{len(popular):,}", f"{len(stats):,}", f"{args.min_ratings:,}")
    if popular.empty:
        raise SystemExit("No movies met the --min-ratings threshold.")

    catalog = movies.merge(popular, on="movie_id", how="inner").sort_values("movie_id").reset_index(drop=True)
    keep_ids = catalog["movie_id"].to_numpy(dtype=np.int32)

    log.info("Step 3/5  Building the rating matrix")
    row_index, user_index, ratings, n_users = load_rating_triples(raw_dir / "rating.csv", keep_ids)

    log.info("Step 4/5  Training item-item collaborative filter")
    neighbour_rows, neighbour_scores = build_similarity(
        row_index, user_index, ratings, len(keep_ids), n_users, args.neighbours
    )
    del row_index, user_index, ratings
    gc.collect()

    log.info("Step 5/5  Attaching keywords, links and posters")
    keywords = load_keywords(raw_dir / "genome_scores.csv", keep_ids, args.keywords)
    catalog = catalog.merge(keywords, on="movie_id", how="left")
    catalog = catalog.merge(load_links(raw_dir / "link.csv"), on="movie_id", how="left")
    catalog["keywords"] = catalog["keywords"].fillna("")
    catalog["imdb_id"] = catalog["imdb_id"].fillna("")
    catalog["tmdb_id"] = catalog["tmdb_id"].fillna("")

    # Preserve any poster URLs a previous build (or scripts/fetch_posters.py)
    # already resolved, so re-running the model does not wipe them.
    catalog["poster_url"] = ""
    existing = out_dir / "movies.csv"
    if existing.exists():
        previous = pd.read_csv(existing, dtype={"movie_id": np.int32}, usecols=lambda c: c in {"movie_id", "poster_url"})
        if "poster_url" in previous.columns:
            merged = catalog.merge(previous, on="movie_id", how="left", suffixes=("", "_old"))
            catalog["poster_url"] = merged["poster_url_old"].fillna("")
            log.info("Carried over %s existing poster URLs", int((catalog["poster_url"] != "").sum()))

    catalog["year"] = catalog["year"].astype("Int64")
    catalog = catalog[[
        "movie_id", "title", "year", "genres", "keywords",
        "rating_count", "rating_avg", "imdb_id", "tmdb_id", "poster_url",
    ]]
    catalog.to_csv(out_dir / "movies.csv", index=False)

    log.info("Building TF-IDF content similarity")
    content_rows, content_scores = build_content_similarity(catalog, args.content_neighbours)
    np.savez_compressed(
        out_dir / "content_similarity.npz",
        movie_ids=keep_ids,
        neighbour_ids=keep_ids[content_rows],
        neighbour_scores=content_scores,
    )

    np.savez_compressed(
        out_dir / "item_similarity.npz",
        movie_ids=keep_ids,
        neighbour_ids=keep_ids[neighbour_rows],
        neighbour_scores=neighbour_scores,
    )

    meta = {
        "algorithm": "hybrid collaborative filtering and content-based filtering",
        "collaborative_algorithm": "item-based collaborative filtering (adjusted cosine similarity)",
        "content_algorithm": "TF-IDF genres and keywords (cosine similarity)",
        "dataset": "MovieLens 20M",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "movies": int(len(catalog)),
        "users": int(n_users),
        "ratings_used": int(catalog["rating_count"].sum()),
        "min_ratings_per_movie": int(args.min_ratings),
        "neighbours_per_movie": int(neighbour_scores.shape[1]),
        "content_neighbours_per_movie": int(content_scores.shape[1]),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    log.info("Done in %.1fs", meta["build_seconds"])
    log.info("  %s  (%s movies)", out_dir / "movies.csv", f"{meta['movies']:,}")
    log.info("  %s  (top-%s neighbours)", out_dir / "item_similarity.npz", meta["neighbours_per_movie"])
    log.info("  %s", out_dir / "model_meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
