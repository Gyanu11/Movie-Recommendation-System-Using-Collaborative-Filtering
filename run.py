"""
Development entry point.
    python run.py
For anything beyond local development, serve the factory through a real WSGI
server instead:
    gunicorn "run:app"
"""
from __future__ import annotations

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=app.config.get("DEBUG", False),
    )
