"""
Flask-WTF forms.
"""
from __future__ import annotations

import re

from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField
from wtforms import (
    BooleanField,
    DecimalField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import (
    URL,
    DataRequired,
    Email,
    EqualTo,
    Length,
    NumberRange,
    Optional,
    ValidationError,
    Regexp,
)

from app.models import MAX_RATING, MIN_RATING, User
from app.services import get_catalog

USERNAME_PATTERN = r"\A[A-Za-z][A-Za-z0-9_]{1,19}\Z"
EMAIL_PATTERN = r"(?i:\A[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@gmail\.com\Z)"
PASSWORD_PATTERN = r"\A(?=.{8,64}\Z)(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9\s])(?!.*\s).+\Z"
MOVIE_ID_PATTERN = r"\A[A-Za-z0-9][A-Za-z0-9_-]{0,49}\Z"
PRINTABLE_TEXT_PATTERN = r"\A(?=.*\S)[^\x00-\x1F\x7F]+\Z"

PASSWORD_MESSAGE = (
    "Password must be 8-64 characters and include an uppercase letter, "
    "a lowercase letter, a number and a special symbol (no spaces)."
)


def _strip(value):
    """Trim text fields before any other validator sees them."""
    return value.strip() if isinstance(value, str) else value


username_validators = [
    DataRequired(),
    Length(min=2, max=20),
    Regexp(USERNAME_PATTERN, message="Username must start with a letter and contain only letters, numbers or underscores."),
]
email_validators = [
    DataRequired(),
    Length(max=120),
    Email(),
    Regexp(EMAIL_PATTERN, message="Email must use the @gmail.com format."),
]
password_validators = [DataRequired(), Regexp(PASSWORD_PATTERN, message=PASSWORD_MESSAGE)]
printable_text = [DataRequired(), Regexp(PRINTABLE_TEXT_PATTERN, message="Enter text without control characters.")]


class _UniqueUserFieldsMixin:
    """Provides reusable username and email uniqueness validation."""

    original_username: str | None = None
    original_email: str | None = None

    def validate_username(self, field):
        if field.data != self.original_username and User.query.filter_by(username=field.data).first():
            raise ValidationError("That username is taken. Please choose a different one.")

    def validate_email(self, field):
        if field.data != self.original_email and User.query.filter_by(email=field.data).first():
            raise ValidationError("That email is already registered.")


class RegistrationForm(FlaskForm, _UniqueUserFieldsMixin):
    username = StringField("Username", validators=username_validators, filters=[_strip])
    email = StringField("Email", validators=email_validators, filters=[_strip])
    password = PasswordField("Password", validators=password_validators)
    confirm_pswd = PasswordField(
        "Confirm Password", validators=[DataRequired(), EqualTo("password", message="Passwords must match.")]
    )
    submit = SubmitField("Sign Up")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=email_validators, filters=[_strip])
    password = PasswordField("Password", validators=[DataRequired()])
    remember = BooleanField("Remember Me")
    submit = SubmitField("Login")


class UpdateAccount(FlaskForm, _UniqueUserFieldsMixin):
    username = StringField("Username", validators=username_validators, filters=[_strip])
    email = StringField("Email", validators=email_validators, filters=[_strip])
    picture = FileField(
        "Update Profile Picture",
        validators=[FileAllowed(["jpg", "jpeg", "png"], "JPG and PNG images only.")],
    )
    submit = SubmitField("Update")

    def __init__(self, original_username=None, original_email=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_username = original_username
        self.original_email = original_email


class MovieForm(FlaskForm):
    """The recommender's single input: a movie the user already likes."""

    moviename = StringField(
        "Movie you like", validators=printable_text + [Length(max=200)], filters=[_strip]
    )
    submit = SubmitField("Get Recommendations")


class RatingForm(FlaskForm):
    """Star rating on MovieLens' own 0.5-5.0 scale."""

    rating = DecimalField(
        "Your rating", places=1, validators=[DataRequired(), NumberRange(min=MIN_RATING, max=MAX_RATING)]
    )
    submit = SubmitField("Save rating")


class AdminMovieForm(FlaskForm):
    """Edits catalog metadata while keeping 20 million MovieLens rating and statistics consistent
    for similarity index."""

    movie_id = StringField(
        "Movie ID",
        validators=[DataRequired(), Regexp(MOVIE_ID_PATTERN, message="Movie ID may contain only letters, numbers, hyphens and underscores.")],
        filters=[_strip],
    )
    title = StringField("Title", validators=printable_text + [Length(max=200)], filters=[_strip])
    year = IntegerField("Year", validators=[DataRequired(), NumberRange(min=1888, max=2100)])
    genres = StringField(
        "Genres (pipe-separated)", validators=printable_text + [Length(max=250)], filters=[_strip]
    )
    keywords = TextAreaField(
        "Keywords (pipe-separated)", validators=[Optional(), Length(max=2000)], filters=[_strip]
    )
    imdb_id = StringField(
        "IMDb ID",
        validators=[Optional(), Regexp(r"\Att\d{7,9}\Z", message="IMDb IDs look like tt0114709.")],
        filters=[_strip],
    )
    poster_url = StringField(
        "Poster URL",
        validators=[Optional(), Length(max=2048), URL(message="Enter a full http(s) URL.")],
        filters=[_strip],
    )
    submit = SubmitField("Save Movie")

    def __init__(self, original_movie_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_movie_id = original_movie_id

    def validate_movie_id(self, field):
        if get_catalog().exists(field.data, exclude_id=self.original_movie_id):
            raise ValidationError("That Movie ID is already in the catalog.")


class AdminUserForm(FlaskForm, _UniqueUserFieldsMixin):
    username = StringField("Username", validators=username_validators, filters=[_strip])
    email = StringField("Email", validators=email_validators, filters=[_strip])
    password = PasswordField("Password")
    is_admin = BooleanField("Administrator")
    submit = SubmitField("Save User")

    def __init__(self, original_username=None, original_email=None, require_password=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_username = original_username
        self.original_email = original_email
        self.require_password = require_password

    def validate_password(self, field):
        # On edit, an empty password means "leave it unchanged".
        if self.require_password and not field.data:
            raise ValidationError("Password is required.")
        if field.data and not re.fullmatch(PASSWORD_PATTERN, field.data):
            raise ValidationError(PASSWORD_MESSAGE)


class AdminWatchlistForm(FlaskForm):
    user_id = SelectField("User", coerce=int, validators=[DataRequired()])
    movie_id = StringField(
        "Movie ID",
        validators=[DataRequired(), Regexp(MOVIE_ID_PATTERN, message="Movie ID may contain only letters, numbers, hyphens, and underscores.")],
        filters=[_strip],
    )
    submit = SubmitField("Add to Watchlist")


class CsrfOnlyForm(FlaskForm):
    """Carries nothing but a CSRF token, for POST-only actions such as delete."""
