import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import load_config, AppConfig
from app.inference import FASInference, NoFaceDetectedError


# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PredictionResponse(BaseModel):
    spoof_score: float

    class Config:
        json_schema_extra = {"example": {"spoof_score": 0.85}}


class ErrorResponse(BaseModel):
    detail: str

    class Config:
        json_schema_extra = {"example": {"detail": "Error message"}}


# Common error responses for Swagger documentation
ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": """**Bad Request** - Invalid input data

Possible causes:
- **Invalid file extension**: File format not supported
- **Empty file**: Uploaded file has no content
- **No face detected**: No valid face found in any frame
""",
        "content": {
            "application/json": {
                "examples": {
                    "invalid_extension": {
                        "summary": "Invalid file extension",
                        "value": {
                            "detail": "Invalid file extension '.txt'. Allowed: .mp4, .avi, .mov, .mkv, .webm"
                        },
                    },
                    "empty_file": {
                        "summary": "Empty file",
                        "value": {"detail": "Empty file"},
                    },
                    "no_face": {
                        "summary": "No face detected",
                        "value": {"detail": "No valid face detected in any frame"},
                    },
                }
            }
        },
    },
    500: {
        "model": ErrorResponse,
        "description": """**Internal Server Error** - Server-side failure

Possible causes:
- **Model inference failed**: GPU/CPU error during prediction
- **Out of memory**: Insufficient memory for processing
""",
        "content": {
            "application/json": {
                "examples": {
                    "inference_failed": {
                        "summary": "Inference failed",
                        "value": {"detail": "Inference failed: CUDA out of memory"},
                    }
                }
            }
        },
    },
}


config: AppConfig | None = None
inference: FASInference | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, inference
    config_path = os.environ.get("FAS_CONFIG", "config.yaml")
    config = load_config(config_path)
    inference = FASInference(config)
    print(f"Model loaded: {config.model.arch} from {config.model.checkpoint}")
    yield
    del inference


app = FastAPI(
    title="Face Anti-Spoofing API",
    description="""
## Overview

This API detects **fake/spoofed faces** in videos and images using deep learning.

## How it works

1. Upload a video or image(s)
2. The API extracts faces and analyzes them
3. Returns a **spoof score** between 0 and 1

## Score Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| **0.0 - 0.4** | Likely **REAL** face |
| **0.4 - 0.6** | Uncertain (review manually) |
| **0.6 - 1.0** | Likely **FAKE/SPOOF** face |

## Recommended Threshold

Use `threshold = 0.6` for classification:
- `spoof_score >= 0.6` → **FAKE**
- `spoof_score < 0.6` → **REAL**
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post(
    "/predict/video",
    response_model=PredictionResponse,
    responses=ERROR_RESPONSES,
    summary="Analyze a video for face spoofing",
    tags=["Prediction"],
)
async def predict_video(
    file: UploadFile = File(..., description="Video file (mp4, avi, mov, etc.)")
):
    """
    Analyze a video file to detect face spoofing.

    The API will:
    1. Extract up to 5 frames from the video
    2. Detect and crop faces from each frame
    3. Run inference on each face
    4. Return the average spoof score

    **Supported formats**: mp4, avi, mov, mkv, webm
    """
    # Check file extension
    if file.filename:
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        spoof_score, _ = inference.predict_video(content)
    except NoFaceDetectedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return PredictionResponse(spoof_score=spoof_score)


@app.post(
    "/predict/images",
    response_model=PredictionResponse,
    responses=ERROR_RESPONSES,
    summary="Analyze image(s) for face spoofing",
    tags=["Prediction"],
)
async def predict_images(
    files: List[UploadFile] = File(..., description="One or more image files (jpg, png, etc.)")
):
    """
    Analyze one or more images to detect face spoofing.

    The API will:
    1. Detect and crop faces from each image
    2. Run inference on each face
    3. Return the average spoof score across all images

    **Supported formats**: jpg, jpeg, png, bmp, webp

    **Tip**: For best results with video content, extract multiple frames
    and submit them together.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    images = []
    for f in files:
        # Check file extension
        if f.filename:
            ext = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file extension '{ext}' for '{f.filename}'. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
                )
        content = await f.read()
        if len(content) > 0:
            images.append(content)

    if not images:
        raise HTTPException(status_code=400, detail="No valid images found")

    try:
        spoof_score, _ = inference.predict_images(images)
    except NoFaceDetectedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    return PredictionResponse(spoof_score=spoof_score)


# Serve demo frontend at /demo (must be after API routes so they take priority)
_demo_dir = Path(__file__).resolve().parent.parent / "demo"
if _demo_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


if __name__ == "__main__":
    import uvicorn

    config_path = os.environ.get("FAS_CONFIG", "config.yaml")
    cfg = load_config(config_path)
    uvicorn.run(
        "app.main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        workers=cfg.server.workers,
    )
