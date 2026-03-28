# server/app.py
# OpenEnv validator requires this path. Imports and re-exports the main app.
from app import app

__all__ = ["app"]
