"""Registration, login, logout and the account page."""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import select

from app.extensions import bcrypt, db
from app.forms import CsrfOnlyForm, LoginForm, RegistrationForm, UpdateAccount
from app.models import User, UserRating, UserWatchlist
from app.services import get_catalog
from app.utils.media import save_profile_picture

auth_bp = Blueprint("auth", __name__)


def _is_safe_next(target: str | None) -> bool:
    """Only follow `?next=` values that stay on this site.

    Without this check the login form is an open redirect: an attacker can send
    `/login?next=https://evil.example` and land a freshly authenticated user
    somewhere hostile.
    """
    if not target:
        return False
    parsed = urlparse(target)
    return not parsed.netloc and not parsed.scheme and target.startswith("/")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=bcrypt.generate_password_hash(form.password.data).decode("utf-8"),
        )
        db.session.add(user)
        db.session.commit()
        flash(f"Account created for {user.username}. You can log in now.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html", title="Register", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.home"))

    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(select(User).where(User.email == form.email.data))
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            flash(f"Welcome back, {user.username}.", "success")

            next_page = request.args.get("next")
            if _is_safe_next(next_page):
                return redirect(next_page)
            return redirect(url_for("admin.dashboard" if user.is_admin else "main.home"))

        # One message for both cases, so the form cannot be used to discover
        # which email addresses are registered.
        flash("Login failed. Check your email and password.", "danger")

    return render_template("login.html", title="Login", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("main.home"))


@auth_bp.route("/account/delete", methods=["POST"])
@login_required
def delete_account():
    form = CsrfOnlyForm()
    if not form.validate_on_submit():
        flash("Your account could not be deleted. Please try again.", "danger")
        return redirect(url_for("auth.account"))

    user = db.session.get(User, current_user.id)
    db.session.delete(user)
    db.session.commit()
    logout_user()
    flash("Your account has been deleted.", "success")
    return redirect(url_for("main.home"))


@auth_bp.route("/account", methods=["GET", "POST"])
@login_required
def account():
    form = UpdateAccount(original_username=current_user.username, original_email=current_user.email)

    if form.validate_on_submit():
        if form.picture.data:
            try:
                current_user.image_file = save_profile_picture(
                    form.picture.data,
                    current_app.config["PROFILE_PIC_FOLDER"],
                    current_app.config["PROFILE_PIC_SIZE"],
                )
            except ValueError as exc:
                flash(str(exc), "danger")
                return redirect(url_for("auth.account"))

        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash("Your account has been updated.", "success")
        return redirect(url_for("auth.account"))

    if request.method == "GET":
        form.username.data = current_user.username
        form.email.data = current_user.email

    catalog = get_catalog()

    watchlist_rows = db.session.scalars(
        select(UserWatchlist)
        .where(UserWatchlist.user_id == current_user.id)
        .order_by(UserWatchlist.added_on.desc())
    ).all()
    rating_rows = db.session.scalars(
        select(UserRating).where(UserRating.user_id == current_user.id).order_by(UserRating.rated_on.desc())
    ).all()

    watchlist = [movie.as_dict() for movie in catalog.get_many(row.movie_id for row in watchlist_rows)]

    rated = []
    for row in rating_rows:
        movie = catalog.get(row.movie_id)
        if movie is None:
            continue
        entry = movie.as_dict()
        entry["your_rating"] = row.rating
        rated.append(entry)

    return render_template(
        "account.html",
        title="Account",
        form=form,
        delete_form=CsrfOnlyForm(),
        image_file=url_for("static", filename=f"profile_pics/{current_user.image_file}"),
        timestamp=int(datetime.now(timezone.utc).timestamp()),
        watchlist=watchlist,
        watchlist_limit=current_app.config["WATCHLIST_MAX_ITEMS"],
        rated=rated,
    )
