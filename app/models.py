"""
Database models.

Only *user* data lives in SQLite. The movie catalog and the trained
similarity model are read-only artifacts loaded from `data/processed/`, so
they are deliberately not modelled as tables.

`UserRating` is what makes the recommendations personal: the collaborative
filter is trained on 20 million MovieLens ratings, and a user's own stars are
the profile that gets matched against it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from flask_login import UserMixin
from sqlalchemy import UniqueConstraint

from app.extensions import db, login_manager

# Ratings a user leaves are stored on MovieLens' own 0.5-5.0 scale so they can
# be compared directly against the trained model without rescaling.
MIN_RATING = 0.5
MAX_RATING = 5.0


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (datetime.utcnow() is deprecated)."""
    return datetime.now(timezone.utc)


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


class User(db.Model, UserMixin):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    image_file = db.Column(db.String(40), nullable=False, default="default.jpg")
    password = db.Column(db.String(60), nullable=False)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

    watchlist_entries = db.relationship(
        "UserWatchlist", back_populates="user", lazy="selectin", cascade="all, delete-orphan"
    )
    ratings = db.relationship(
        "UserRating", back_populates="user", lazy="selectin", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.username!r}>"


class UserWatchlist(db.Model):
    __tablename__ = "user_watchlist"
    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_watchlist_user_movie"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = db.Column(db.String(50), nullable=False, index=True)
    added_on = db.Column(db.DateTime, nullable=False, default=_utcnow)

    user = db.relationship("User", back_populates="watchlist_entries")

    def __repr__(self) -> str:
        return f"<UserWatchlist user={self.user_id} movie={self.movie_id}>"


class UserRating(db.Model):
    __tablename__ = "user_rating"
    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_rating_user_movie"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = db.Column(db.String(50), nullable=False, index=True)
    rating = db.Column(db.Float, nullable=False)
    rated_on = db.Column(db.DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)

    user = db.relationship("User", back_populates="ratings")

    def __repr__(self) -> str:
        return f"<UserRating user={self.user_id} movie={self.movie_id} rating={self.rating}>"
