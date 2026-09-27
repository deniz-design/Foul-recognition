"""Central configuration for the backend.

Every path is resolved relative to this file rather than the current working
directory. Previously the code hardcoded paths like ``./backend/pretrained_models``
while the README told you to run ``cd backend && python app.py``, so the paths
only worked from the repository root. Resolving against ``__file__`` makes the
app work from any directory.

All values can be overridden with environment variables.
"""

import os
from pathlib import Path

# backend/
BACKEND_DIR = Path(__file__).resolve().parent
# repository root (the parent of backend/)
PROJECT_ROOT = BACKEND_DIR.parent


def _env_path(name: str, default: Path) -> Path:
    """Resolve an env var into an absolute path, falling back to ``default``."""
    raw = os.getenv(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return default


# --- Paths -----------------------------------------------------------------

# Where trained checkpoints are written by the training pipeline.
PRETRAINED_DIR = BACKEND_DIR / "pretrained_models"

# The checkpoint the project was originally developed against. It is not
# distributed, so the lookup below falls back to the newest checkpoint present.
DEFAULT_MODEL_RELPATH = Path("pretrained_models/20250129_195031/foul_detection_model.pth")
DEFAULT_MODEL_PATH = BACKEND_DIR / DEFAULT_MODEL_RELPATH


def discover_model_path() -> Path:
    """Resolve which checkpoint to load.

    Precedence:
      1. FOUL_MODEL_PATH, if set.
      2. The original default path, if present.
      3. The most recently modified foul_detection_model.pth under
         pretrained_models/, so a freshly trained model is picked up without
         having to export an environment variable.
    """
    explicit = os.getenv("FOUL_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    if PRETRAINED_DIR.is_dir():
        candidates = sorted(
            PRETRAINED_DIR.glob("*/foul_detection_model.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

    return DEFAULT_MODEL_PATH


MODEL_PATH = discover_model_path()

# Where the SoccerNet clips are extracted to (mvfouls-download.py writes here).
DATA_ROOT = _env_path("FOUL_DATA_ROOT", BACKEND_DIR / "data")

# Where per-split annotations and extracted .h5 features live.
BASE_DIR = _env_path("FOUL_BASE_DIR", DATA_ROOT / "dataset")
INFERENCE_DIR = BASE_DIR / "inference"

# --- Server ----------------------------------------------------------------

HOST = os.getenv("HOST", "127.0.0.1")
PORT = _env_int("PORT", 5000)

# Comma-separated list of allowed origins. "*" disables the CORS middleware
# entirely. We no longer pair allow_origins=["*"] with allow_credentials=True,
# which browsers reject as an invalid combination.
# Both spellings of the frontend host are allowed because browsers treat
# http://localhost:3000 and http://127.0.0.1:3000 as different origins.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]

# --- Upload limits ---------------------------------------------------------

MAX_CONTENT_LENGTH = _env_int("MAX_CONTENT_LENGTH_MB", 100) * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "wmv", "mkv", "webm"}

# The frame count the r3d_18 backbone was trained with. Do not change: the
# model is sensitive to clip length.
NUM_FRAMES = 25
FRAME_SIZE = 224
FEATURE_EXTRACTOR_MODEL = os.getenv("FEATURE_MODEL_TYPE", "r3d_18")

# Set to false to hard-fail instead of serving with untrained weights.
ALLOW_UNTRAINED_FALLBACK = _env_bool("ALLOW_UNTRAINED_FALLBACK", True)

# Deterministic weights when running without a checkpoint.
FALLBACK_SEED = _env_int("FALLBACK_SEED", 0)
