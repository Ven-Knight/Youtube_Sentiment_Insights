import os
from dotenv import load_dotenv

# Load environment variables from .env inside flask_api/
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "flask_api", ".env"))

# ─────────────────────────────────────────────────────────────────────────────
# External Service URIs with defaults
# ─────────────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
API_PORT            = int(os.getenv("FLASK_PORT", 8080))  # Port Flask will run on

# ─────────────────────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY     = os.getenv("YOUTUBE_API_KEY")