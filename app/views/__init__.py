"""
Route modules, split by area and wired up as Flask blueprints.

Blueprints namespace their endpoints, so templates refer to
`url_for('main.home')`, `url_for('auth.login')` and `url_for('admin.movies')`.
That makes it obvious at a glance which part of the app a link points at, and
lets the admin area carry a single `/admin` URL prefix instead of repeating it
on every route.
"""
from __future__ import annotations

from flask import Flask

from app.views.admin import admin_bp
from app.views.auth import auth_bp
from app.views.main import main_bp


def register_all(app: Flask) -> None:
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
