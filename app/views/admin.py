"""Admin console: catalog, user and watchlist management."""
from __future__ import annotations

from math import ceil

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from app.decorators import admin_required
from app.extensions import bcrypt, db
from app.forms import AdminMovieForm, AdminUserForm, AdminWatchlistForm, CsrfOnlyForm
from app.models import User, UserRating, UserWatchlist
from app.services import Movie, get_catalog, get_engine
from app.utils.text import normalise

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def _movie_from_form(form: AdminMovieForm) -> Movie:
    return Movie(
        movie_id=form.movie_id.data,
        title=form.title.data,
        year=str(form.year.data or ""),
        genres=form.genres.data,
        keywords=form.keywords.data or "",
        imdb_id=form.imdb_id.data or "",
        poster_url=form.poster_url.data or "",
    )


def _reject_without_csrf(redirect_endpoint: str):
    """Validate a bare CSRF form; return a redirect response on failure."""
    if CsrfOnlyForm().validate_on_submit():
        return None
    flash("That request could not be verified. Please try again.", "danger")
    return redirect(url_for(redirect_endpoint))


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
@admin_bp.route("/")
@admin_required
def dashboard():
    catalog = get_catalog()
    engine = get_engine()

    # Counting in SQL avoids loading every row just to call len() on it.
    user_count = db.session.scalar(select(func.count()).select_from(User)) or 0
    admin_count = db.session.scalar(select(func.count()).select_from(User).where(User.is_admin)) or 0
    watchlist_count = db.session.scalar(select(func.count()).select_from(UserWatchlist)) or 0
    rating_count = db.session.scalar(select(func.count()).select_from(UserRating)) or 0

    recent_users = db.session.scalars(select(User).order_by(User.id.desc()).limit(5)).all()

    return render_template(
        "admin_dashboard.html",
        title="Admin Dashboard",
        movie_count=len(catalog),
        user_count=user_count,
        admin_count=admin_count,
        watchlist_count=watchlist_count,
        rating_count=rating_count,
        recent_users=recent_users,
        top_movies=[movie.as_dict() for movie in catalog.most_popular(5)],
        model_meta=engine.index.meta,
        model_ready=engine.index.is_ready,
    )


# --------------------------------------------------------------------------- #
# Movies
# --------------------------------------------------------------------------- #
@admin_bp.route("/movies")
@admin_required
def movies():
    catalog = get_catalog()
    query = request.args.get("q", "").strip()

    try:
        page = max(int(request.args.get("page", 1)), 1)
    except (TypeError, ValueError):
        page = 1

    rows = catalog.all_movies()
    if query:
        needle = normalise(query)
        rows = [
            movie
            for movie in rows
            if needle in movie.normalised_title
            or query == movie.movie_id
            or needle in normalise(movie.genres)
        ]

    per_page = current_app.config["ADMIN_PAGE_SIZE"]
    total = len(rows)
    pages = max(ceil(total / per_page), 1)
    page = min(page, pages)
    start = (page - 1) * per_page

    return render_template(
        "admin_movies.html",
        title="Manage Movies",
        movies=[movie.as_dict() for movie in rows[start : start + per_page]],
        query=query,
        page=page,
        pages=pages,
        total=total,
        csrf_form=CsrfOnlyForm(),
    )


@admin_bp.route("/movies/new", methods=["GET", "POST"])
@admin_required
def movie_create():
    catalog = get_catalog()
    form = AdminMovieForm()

    if request.method == "GET" and not form.movie_id.data:
        form.movie_id.data = catalog.next_movie_id()

    if form.validate_on_submit():
        try:
            catalog.create(_movie_from_form(form))
        except (ValueError, OSError) as exc:
            flash(str(exc), "danger")
        else:
            flash(
                "Movie added. Note that titles added by hand have no MovieLens "
                "rating history, so they cannot seed recommendations until the "
                "model is rebuilt.",
                "info",
            )
            return redirect(url_for("admin.movies"))

    return render_template("admin_movie_form.html", title="Add Movie", form=form, mode="create")


@admin_bp.route("/movies/<movie_id>/edit", methods=["GET", "POST"])
@admin_required
def movie_edit(movie_id):
    catalog = get_catalog()
    movie = catalog.get(movie_id)
    if movie is None:
        flash("Movie not found.", "warning")
        return redirect(url_for("admin.movies"))

    form = AdminMovieForm(original_movie_id=movie.movie_id)

    if form.validate_on_submit():
        try:
            updated = catalog.update(movie.movie_id, _movie_from_form(form))
        except (ValueError, OSError) as exc:
            flash(str(exc), "danger")
        else:
            if updated.movie_id != movie.movie_id:
                # Keep users' saved rows pointing at the same film.
                for model in (UserWatchlist, UserRating):
                    db.session.query(model).filter_by(movie_id=movie.movie_id).update(
                        {"movie_id": updated.movie_id}
                    )
                db.session.commit()
            flash("Movie updated.", "success")
            return redirect(url_for("admin.movies"))

    elif request.method == "GET":
        form.movie_id.data = movie.movie_id
        form.title.data = movie.title
        form.genres.data = movie.genres
        form.keywords.data = movie.keywords
        form.imdb_id.data = movie.imdb_id
        form.poster_url.data = movie.poster_url
        form.year.data = int(movie.year) if str(movie.year).isdigit() else None

    return render_template(
        "admin_movie_form.html", title="Edit Movie", form=form, mode="edit", movie=movie.as_dict()
    )


@admin_bp.route("/movies/<movie_id>/delete", methods=["POST"])
@admin_required
def movie_delete(movie_id):
    rejected = _reject_without_csrf("admin.movies")
    if rejected:
        return rejected

    key = str(movie_id).strip()
    if get_catalog().delete(key):
        for model in (UserWatchlist, UserRating):
            db.session.execute(delete(model).where(model.movie_id == key))
        db.session.commit()
        flash("Movie deleted from the catalog.", "success")
    else:
        flash("Movie not found.", "warning")

    return redirect(url_for("admin.movies"))


# --------------------------------------------------------------------------- #
# Users
# --------------------------------------------------------------------------- #
@admin_bp.route("/users")
@admin_required
def users():
    rows = db.session.scalars(select(User).order_by(User.id.asc())).all()
    return render_template("admin_users.html", title="Manage Users", users=rows, csrf_form=CsrfOnlyForm())


@admin_bp.route("/users/new", methods=["GET", "POST"])
@admin_required
def user_create():
    form = AdminUserForm(require_password=True)
    if form.validate_on_submit():
        db.session.add(
            User(
                username=form.username.data,
                email=form.email.data,
                password=bcrypt.generate_password_hash(form.password.data).decode("utf-8"),
                is_admin=bool(form.is_admin.data),
            )
        )
        db.session.commit()
        flash("User created.", "success")
        return redirect(url_for("admin.users"))
    return render_template("admin_user_form.html", title="Add User", form=form, mode="create")


@admin_bp.route("/users/<int:user_id>/edit", methods=["GET", "POST"])
@admin_required
def user_edit(user_id):
    user = db.get_or_404(User, user_id)
    form = AdminUserForm(
        original_username=user.username, original_email=user.email, require_password=False
    )

    if form.validate_on_submit():
        # Removing the last administrator would lock everyone out of /admin.
        if user.is_admin and not form.is_admin.data and _admin_count() <= 1:
            flash("You cannot remove the last administrator.", "warning")
            return redirect(url_for("admin.user_edit", user_id=user.id))

        user.username = form.username.data
        user.email = form.email.data
        user.is_admin = bool(form.is_admin.data)
        if form.password.data:
            user.password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        db.session.commit()
        flash("User updated.", "success")
        return redirect(url_for("admin.users"))

    if request.method == "GET":
        form.username.data = user.username
        form.email.data = user.email
        form.is_admin.data = bool(user.is_admin)

    return render_template("admin_user_form.html", title="Edit User", form=form, mode="edit", user=user)


@admin_bp.route("/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def user_delete(user_id):
    rejected = _reject_without_csrf("admin.users")
    if rejected:
        return rejected

    user = db.get_or_404(User, user_id)
    if user.id == current_user.id:
        flash("You cannot delete the account you are logged in with.", "warning")
        return redirect(url_for("admin.users"))
    if user.is_admin and _admin_count() <= 1:
        flash("You cannot delete the last administrator.", "warning")
        return redirect(url_for("admin.users"))

    # Watchlist and rating rows cascade via the relationship definitions.
    db.session.delete(user)
    db.session.commit()
    flash("User deleted.", "success")
    return redirect(url_for("admin.users"))


def _admin_count() -> int:
    return db.session.scalar(select(func.count()).select_from(User).where(User.is_admin)) or 0


# --------------------------------------------------------------------------- #
# Watchlists
# --------------------------------------------------------------------------- #
@admin_bp.route("/watchlist", methods=["GET", "POST"])
@admin_required
def watchlist():
    catalog = get_catalog()
    form = AdminWatchlistForm()
    form.user_id.choices = [
        (user.id, f"{user.username} ({user.email})")
        for user in db.session.scalars(select(User).order_by(User.username)).all()
    ]

    if form.validate_on_submit():
        movie = catalog.get(form.movie_id.data)
        if movie is None:
            flash("That Movie ID is not in the catalog.", "warning")
        else:
            db.session.add(UserWatchlist(user_id=form.user_id.data, movie_id=movie.movie_id))
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash("That movie is already in the user's watchlist.", "info")
            else:
                flash("Watchlist entry added.", "success")
                return redirect(url_for("admin.watchlist"))

    rows = db.session.execute(
        select(UserWatchlist, User)
        .join(User, User.id == UserWatchlist.user_id)
        .order_by(UserWatchlist.added_on.desc())
    ).all()

    entries = [
        {
            "id": entry.id,
            "username": user.username,
            "email": user.email,
            "movie_id": entry.movie_id,
            "title": (catalog.get(entry.movie_id) or None) and catalog.get(entry.movie_id).title,
            "added_on": entry.added_on,
        }
        for entry, user in rows
    ]

    return render_template(
        "admin_watchlist.html",
        title="Manage Watchlists",
        form=form,
        entries=entries,
        csrf_form=CsrfOnlyForm(),
    )


@admin_bp.route("/watchlist/<int:entry_id>/delete", methods=["POST"])
@admin_required
def watchlist_delete(entry_id):
    rejected = _reject_without_csrf("admin.watchlist")
    if rejected:
        return rejected

    entry = db.get_or_404(UserWatchlist, entry_id)
    db.session.delete(entry)
    db.session.commit()
    flash("Watchlist entry deleted.", "success")
    return redirect(url_for("admin.watchlist"))
