import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables from .env file for local development
# ─────────────────────────────────────────────────────────────────────────────
# This ensures that when running locally (outside Docker), config values like
# MLFLOW_TRACKING_URI and YOUTUBE_API_KEY are available via os.environ.

# In production or CI/CD, these variables are injected at runtime and this
# loading step is safely ignored.
# ─────────────────────────────────────────────────────────────────────────────

# Resolve path to .env file inside flask_api folder
dotenv_path = os.path.join(os.path.dirname(__file__), "flask_api", ".env")
load_dotenv(dotenv_path=dotenv_path)

# ─────────────────────────────────────────────────────────────────────────────
# Centralized runtime configuration
# ─────────────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
YOUTUBE_API_KEY     = os.environ.get("YOUTUBE_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# - Local runs   : .env is loaded explicitly
# - Docker/CI/CD : .env is ignored, and values come from -e injection
# - No code changes needed between environments
# ─────────────────────────────────────────────────────────────────────────────
