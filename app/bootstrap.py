"""
First-run setup that creates missing tables, ensures a default admin account
and removes outdated watchlist/rating records after catalog changes.
"""

from __future__ import annotations

import logging

from flask import Flask
from sqlalchemy import inspect, select, text

from app.extensions import bcrypt, db
from app.models import User, UserRating, UserWatchlist

log = logging.getLogger(__name__)


def init_app_data(app: Flask) -> None:
    with app.app_context():
        db.create_all()
        _ensure_user_is_active_column()
        _ensure_admin_account(
            app.config["ADMIN_USERNAME"],
            app.config["ADMIN_EMAIL"],
            app.config["ADMIN_PASSWORD"],
        )
        _prune_orphaned_rows(app)


def _ensure_user_is_active_column() -> None:
    """create_all() never alters existing tables, so add `is_active` to older databases."""
    columns = {col["name"] for col in inspect(db.engine).get_columns("user")}
    if "is_active" in columns:
        return
    db.session.execute(text('ALTER TABLE "user" ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT 1'))
    db.session.commit()
    log.info("Added is_active column to the user table")


def _ensure_admin_account(username: str, email: str, password: str) -> None:
    """Create the default admin, or promote it if the account already exists."""
    existing = db.session.scalar(
        select(User).where((User.username == username) | (User.email == email))
    )

    if existing is not None:
        if not existing.is_admin or not existing.is_active:
            existing.is_admin = True
            existing.is_active = True
            db.session.commit()
            log.info("Promoted existing account %r to administrator", existing.username)
        return

    db.session.add(
        User(
            username=username,
            email=email,
            password=bcrypt.generate_password_hash(password).decode("utf-8"),
            is_admin=True,
        )
    )
    db.session.commit()
    log.info("Created default administrator %r (%s)", username, email)


def _prune_orphaned_rows(app: Flask) -> None:
    """Delete watchlist and rating rows whose movie is not in the catalog."""
    catalog = app.extensions.get("movie_catalog")
    if catalog is None or len(catalog) == 0:
        return  # catalog failed to load; deleting rows now would be destructive

    valid_ids = {movie.movie_id for movie in catalog.all_movies()}
    removed = 0

    for model in (UserWatchlist, UserRating):
        for row in db.session.scalars(select(model)).all():
            if row.movie_id not in valid_ids:
                db.session.delete(row)
                removed += 1

    if removed:
        db.session.commit()
        log.info("Removed %s saved rows referencing movies outside the current catalog", removed)
