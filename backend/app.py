"""FastAPI application exposing the foul-recognition inference endpoint."""

import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Make sibling modules (config, inference, models, utils) importable regardless
# of the working directory. This must happen before `import config` below, so it
# also works for `python -m backend.app` from the repository root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn  # noqa: E402
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import config  # noqa: E402
from inference import InferencePipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Raised when the video cannot be decoded or featurised."""


class PredictionDetail(BaseModel):
    prediction: str
    probability: float


class CategoryResult(BaseModel):
    category: str
    details: List[PredictionDetail]


class PredictionResponse(BaseModel):
    status: str
    predictions: List[CategoryResult]
    ai_decision: Optional[str] = None
    explanation: Optional[str] = None
    model_trained: bool = True


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_trained: bool
    device: str
    model_path: str
    model_path_exists: bool
    azure_openai_configured: bool


app = FastAPI(
    title="AI Football Referee API",
    description="Analyses football incident clips and returns a referee decision.",
    version="1.1.0",
)

# Only install CORS when explicit origins are configured. Pairing a wildcard
# origin with allow_credentials=True is rejected by browsers, so the previous
# allow_origins=["*"] + allow_credentials=True combination could not have
# worked for credentialed requests.
if config.CORS_ORIGINS and "*" not in config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    logger.info("CORS origins: %s", ", ".join(config.CORS_ORIGINS))
else:
    logger.info("CORS middleware disabled (CORS_ORIGINS contains '*' or is empty)")


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


def validate_video_file(file: UploadFile) -> None:
    """Validate the uploaded video file."""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type. Allowed types: "
                f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
            )
        )


def format_predictions(raw_predictions: List[Dict[str, Any]]) -> List[CategoryResult]:
    """Format raw model predictions into the API response shape."""
    formatted = []
    for pred in raw_predictions:
        formatted.append(CategoryResult(
            category=pred['category'],
            details=[PredictionDetail(
                prediction=str(pred['prediction']),
                probability=float(pred['probability'])
            )]
        ))

        confidence = "#" * int(pred['probability'] * 20) + "-" * (20 - int(pred['probability'] * 20))
        logger.info("%s -> %s [%s] %.1f%%",
                    pred['category'], pred['prediction'], confidence, pred['probability'] * 100)

    return formatted


def cleanup_files(*files: Optional[str]) -> None:
    """Clean up temporary files, ignoring anything already gone."""
    for file_path in files:
        if not file_path:
            continue
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info("Cleaned up file: %s", file_path)
        except Exception as exc:
            logger.error("Error cleaning up %s: %s", file_path, exc)


async def save_upload(video_file: UploadFile) -> str:
    """Stream an upload to a temp file, enforcing the size limit."""
    suffix = os.path.splitext(video_file.filename or "")[1] or '.mp4'
    total = 0
    limit = config.MAX_CONTENT_LENGTH

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = temp_file.name
        while chunk := await video_file.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                break
            temp_file.write(chunk)

    if total > limit:
        cleanup_files(temp_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {limit // (1024 * 1024)}MB"
        )

    if total == 0:
        cleanup_files(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return temp_path


# Initialise the pipeline once at import. A missing checkpoint no longer aborts
# startup: InferencePipeline falls back to randomly initialised weights unless
# ALLOW_UNTRAINED_FALLBACK is disabled.
try:
    pipeline = InferencePipeline(
        model_path=str(config.MODEL_PATH),
        base_dir=str(config.BASE_DIR),
    )
except Exception:
    logger.critical("Failed to initialise the inference pipeline.", exc_info=True)
    raise


@app.get("/", tags=["Meta"])
async def root():
    return {
        "service": "AI Football Referee API",
        "docs": "/docs",
        "health": "/api/health",
        "inference": "POST /api/inference",
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Report service state, including whether real weights are loaded."""
    from explanation import REQUIRED_AZURE_VARS

    return HealthResponse(
        status="ok",
        model_trained=bool(getattr(pipeline, "weights_loaded", False)),
        device=str(pipeline.device),
        model_path=str(config.MODEL_PATH),
        model_path_exists=config.MODEL_PATH.exists(),
        azure_openai_configured=all(os.getenv(v) for v in REQUIRED_AZURE_VARS),
    )


@app.post(
    "/api/inference",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Inference"],
)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    """Process a video file and return predictions with a referee decision."""
    temp_path: Optional[str] = None
    features_path: Optional[str] = None

    try:
        validate_video_file(video)
        temp_path = await save_upload(video)

        try:
            logger.info("Processing video for inference...")
            features_path = pipeline.process_video_for_inference(temp_path, replay_speed=1.0)

            logger.info("Running inference...")
            raw_predictions = pipeline.inference(features_path)
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            # The upload was readable but could not be decoded: corrupt,
            # truncated, or a container OpenCV cannot handle. That is bad input
            # from the client, not a server fault.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Error processing video: %s", exc, exc_info=True)
            raise VideoProcessingError(str(exc)) from exc

        predictions = format_predictions(raw_predictions)

        try:
            from explanation import generate_explanation
            explanation_result = generate_explanation(predictions)
        except Exception as exc:
            logger.error("Explanation generation failed: %s", exc)
            explanation_result = {
                "decision": "Unknown",
                "explanation": "No explanation available",
            }

        logger.info("Inference completed successfully!")

        return PredictionResponse(
            status="success",
            predictions=predictions,
            ai_decision=explanation_result.get("decision", "Unknown"),
            explanation=explanation_result.get("explanation", "No explanation available"),
            model_trained=bool(getattr(pipeline, "weights_loaded", False)),
        )

    except HTTPException:
        raise
    except VideoProcessingError as exc:
        raise HTTPException(status_code=500, detail=f"Error processing video: {exc}") from exc
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred") from exc
    finally:
        # Always remove the uploaded clip and the intermediate feature file.
        cleanup_files(temp_path, features_path)


def main() -> None:
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"},
    )


if __name__ == '__main__':
    main()
