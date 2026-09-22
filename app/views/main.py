"""Public browsing, search, recommendations, watchlist and ratings."""

from __future__ import annotations

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.forms import MovieForm
from app.models import MAX_RATING, MIN_RATING, UserRating, UserWatchlist
from app.services import ColdStartError, get_catalog, get_engine, to_dicts
from app.utils.posters import placeholder_svg
from app.utils.youtube import resolve_youtube_trailer

main_bp = Blueprint("main", __name__)

# Browsers cache generated posters aggressively; they only change if a movie's
# id changes, which never happens for a given catalog entry.
_POSTER_CACHE_SECONDS = 60 * 60 * 24 * 30


# Helpers
def _safe_redirect(fallback_endpoint: str = "main.home"):
    """Safely redirect the user to a same-site URL or a trusted fallback."""
    referrer = request.referrer or ""
    if referrer.startswith(request.host_url):
        return redirect(referrer)
    return redirect(url_for(fallback_endpoint))


def _user_ratings(user_id: int) -> dict[str, float]:
    rows = db.session.scalars(select(UserRating).where(UserRating.user_id == user_id)).all()
    return {row.movie_id: row.rating for row in rows}


def _user_watchlist_ids(user_id: int) -> list[str]:
    rows = db.session.scalars(
        select(UserWatchlist).where(UserWatchlist.user_id == user_id).order_by(UserWatchlist.added_on.desc())
    ).all()
    return [row.movie_id for row in rows]


def _personal_recommendations(user_id: int, limit: int) -> tuple[list[dict], bool]:
    """Recommendations for one user, plus whether they are a popularity fallback."""
    engine = get_engine()
    ratings = _user_ratings(user_id)
    watchlist = _user_watchlist_ids(user_id)

    results = engine.for_user(ratings, watchlist, limit=limit)
    if results:
        return to_dicts(results), False

    seen = set(ratings) | set(watchlist)
    popular = get_catalog().most_popular(limit, exclude=seen)
    return [movie.as_dict() for movie in popular], True


# Browsing
@main_bp.route("/")
@main_bp.route("/home")
def home():
    catalog = get_catalog()
    count = current_app.config["HOMEPAGE_FEATURED_COUNT"]

    personal, is_fallback = ([], True)
    if current_user.is_authenticated:
        personal, is_fallback = _personal_recommendations(
            current_user.id, current_app.config["PERSONAL_RESULT_LIMIT"]
        )

    return render_template(
        "home.html",
        featured=[movie.as_dict() for movie in catalog.top_rated(count)],
        trending=[movie.as_dict() for movie in catalog.most_popular(count)],
        personal=personal,
        personal_is_fallback=is_fallback,
    )


@main_bp.route("/about")
def about():
    engine = get_engine()
    return render_template("about.html", title="About", meta=engine.index.meta, catalog_size=len(get_catalog()))


@main_bp.route("/movie/<movie_id>")
def movie_info(movie_id):
    catalog = get_catalog()
    movie = catalog.get(movie_id)
    if movie is None:
        abort(404)

    similar = to_dicts(get_engine().similar_to(movie.movie_id, limit=10))

    user_rating = None
    in_watchlist = False
    if current_user.is_authenticated:
        row = db.session.scalar(
            select(UserRating).where(
                UserRating.user_id == current_user.id, UserRating.movie_id == movie.movie_id
            )
        )
        user_rating = row.rating if row else None
        in_watchlist = db.session.scalar(
            select(UserWatchlist).where(
                UserWatchlist.user_id == current_user.id, UserWatchlist.movie_id == movie.movie_id
            )
        ) is not None

    return render_template(
        "movieinfo.html",
        title=movie.title,
        movie=movie.as_dict(),
        similar=similar,
        user_rating=user_rating,
        in_watchlist=in_watchlist,
    )


@main_bp.route("/surprise")
@login_required
def surprise():
    movie = get_catalog().random_movie()
    if movie is None:
        flash("The catalog is empty. Run scripts/build_model.py to populate it.", "warning")
        return redirect(url_for("main.home"))
    return redirect(url_for("main.movie_info", movie_id=movie.movie_id))


@main_bp.route("/search")
def search():
    query = request.args.get("q", "").strip()
    minimum = current_app.config["SEARCH_MIN_QUERY_LENGTH"]

    results = []
    if len(query) >= minimum:
        matches = get_catalog().search(query, limit=current_app.config["SEARCH_RESULT_LIMIT"])
        results = [movie.as_dict() for movie in matches]

    return render_template(
        "search_results.html", title=f"Search: {query}" if query else "Search",
        results=results, query=query, min_length=minimum,
    )


# Recommendations
@main_bp.route("/recommender", methods=["GET", "POST"])
@login_required
def recommender():
    form = MovieForm()
    seed = None
    results = []

    if form.validate_on_submit():
        try:
            match, recommendations = get_engine().recommend_by_title(
                form.moviename.data, limit=current_app.config["RECOMMENDER_RESULT_LIMIT"]
            )
        except ColdStartError as exc:
            seed = exc.movie.as_dict()
            flash(str(exc), "warning")
        except LookupError as exc:
            flash(str(exc), "danger")
        else:
            seed = match.as_dict()
            results = to_dicts(recommendations)

    return render_template("recommender.html", title="Recommender", form=form, seed=seed, results=results)


@main_bp.route("/for-you")
@login_required
def for_you():
    recommendations, is_fallback = _personal_recommendations(
        current_user.id, current_app.config["PERSONAL_RESULT_LIMIT"]
    )
    return render_template(
        "for_you.html",
        title="For you",
        results=recommendations,
        is_fallback=is_fallback,
        rated_count=len(_user_ratings(current_user.id)),
    )


@main_bp.route("/api/suggest")
def suggest():
    """Autocomplete feed for the recommender input."""
    query = request.args.get("q", "").strip()
    matches = get_catalog().suggest(query, limit=8)
    return jsonify([{"movie_id": m.movie_id, "title": m.title, "year": m.year} for m in matches])


# Watchlist and ratings
@main_bp.route("/watchlist/add/<movie_id>", methods=["POST"])
@login_required
def add_to_watchlist(movie_id):
    if get_catalog().get(movie_id) is None:
        flash("That movie is not in the catalog.", "warning")
        return _safe_redirect()

    limit = current_app.config["WATCHLIST_MAX_ITEMS"]
    current = len(_user_watchlist_ids(current_user.id))
    if current >= limit:
        flash(f"Your watchlist is full ({limit} movies). Remove one to add another.", "warning")
        return _safe_redirect("auth.account")

    db.session.add(UserWatchlist(user_id=current_user.id, movie_id=str(movie_id).strip()))
    try:
        db.session.commit()
    except IntegrityError:
        # The unique constraint is the source of truth; a duplicate click here
        # is expected, not exceptional.
        db.session.rollback()
        flash("That movie is already in your watchlist.", "info")
    else:
        flash("Added to your watchlist.", "success")
    return _safe_redirect("auth.account")


@main_bp.route("/watchlist/remove/<movie_id>", methods=["POST"])
@login_required
def remove_from_watchlist(movie_id):
    entry = db.session.scalar(
        select(UserWatchlist).where(
            UserWatchlist.user_id == current_user.id, UserWatchlist.movie_id == str(movie_id).strip()
        )
    )
    if entry is None:
        flash("That movie is not in your watchlist.", "warning")
    else:
        db.session.delete(entry)
        db.session.commit()
        flash("Removed from your watchlist.", "success")
    return _safe_redirect("auth.account")


@main_bp.route("/rate/<movie_id>", methods=["POST"])
@login_required
def rate_movie(movie_id):
    """Record or update a star rating, which feeds the personal recommender."""
    if get_catalog().get(movie_id) is None:
        flash("That movie is not in the catalog.", "warning")
        return _safe_redirect()

    raw = request.form.get("rating", "")
    try:
        rating = round(float(raw) * 2) / 2  # snap to the nearest half star
    except (TypeError, ValueError):
        flash("Please choose a rating between 0.5 and 5.", "danger")
        return _safe_redirect()

    key = str(movie_id).strip()
    existing = db.session.scalar(
        select(UserRating).where(UserRating.user_id == current_user.id, UserRating.movie_id == key)
    )

    if rating <= 0:  # a zero submission means "clear my rating"
        if existing:
            db.session.delete(existing)
            db.session.commit()
            flash("Rating removed.", "info")
        return _safe_redirect()

    if not MIN_RATING <= rating <= MAX_RATING:
        flash(f"Ratings must be between {MIN_RATING} and {MAX_RATING}.", "danger")
        return _safe_redirect()

    if existing:
        existing.rating = rating
    else:
        db.session.add(UserRating(user_id=current_user.id, movie_id=key, rating=rating))
    db.session.commit()

    flash(f"Rated {rating:g} out of 5. Your recommendations have been updated.", "success")
    return _safe_redirect()


# Media helpers
@main_bp.route("/poster/<movie_id>.svg")
def poster_placeholder(movie_id):
    """Generated poster artwork for movies with no image URL."""
    movie = get_catalog().get(movie_id)
    if movie is None:
        abort(404)

    svg = placeholder_svg(movie.movie_id, movie.title, movie.year, movie.genres)
    response = current_app.response_class(svg, mimetype="image/svg+xml")
    response.headers["Cache-Control"] = f"public, max-age={_POSTER_CACHE_SECONDS}, immutable"
    return response


@main_bp.route("/trailer")
def trailer():
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"embed_url": "", "error": "A movie title is required."}), 400
    year = request.args.get("year", "").strip()
    return jsonify({"embed_url": resolve_youtube_trailer(title, year)})
