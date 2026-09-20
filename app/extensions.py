"""
Shared extension instances.

They live in their own module (rather than in `app/__init__.py`) so that
models, forms, and views can import them without importing the application
factory, which would create a circular import.
"""
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to continue."
login_manager.login_message_category = "info"
