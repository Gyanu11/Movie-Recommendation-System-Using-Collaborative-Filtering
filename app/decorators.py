"""Reusable route decorators."""
from functools import wraps

from flask import flash, redirect, url_for
from flask_login import current_user, login_required


def admin_required(view):
    """Restrict a view to authenticated users whose `is_admin` flag is set."""

    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not getattr(current_user, "is_admin", False):
            flash("Only administrators can access that page.", "danger")
            return redirect(url_for("main.home"))
        return view(*args, **kwargs)

    return wrapped
