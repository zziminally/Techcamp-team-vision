# CLAUDE.md

## Project Overview

Live Face Anti-Spoofing (FAS) demo for Techcamp 2026 (Team Vision). A single repo containing both the FastAPI backend (model inference) and a browser-based frontend that accesses the webcam in real time, sends video clips for prediction, and visually alerts when spoofing (replay attack, print attack) is detected.

## Repo Structure

- `app/` — FastAPI backend (main.py, inference.py, config.py, preprocess.py)
- `nets/` — Neural network architectures (ResNet, MobileNet, Swin, etc.)
- `scripts/` — Training and dataset preparation scripts
- `checkpoints/` — Model weights (gitignored, 180MB .pth files)
- `demo/` — Frontend demo page (vanilla HTML/CSS/JS)
- `config.yaml` — Model, face detection, video, and server configuration

## Running

```bash
# Install dependencies
uv sync --extra mps   # macOS with Apple Silicon
uv sync --extra gpu   # Linux/Windows with CUDA

# Start the server (serves API + demo frontend)
uv run python -m app.main

# API docs: http://localhost:8000/docs
# Demo page: http://localhost:8000/demo/
```

## API Contract

- `POST /predict/video` — multipart form data, `file` field with video (.webm/.mp4) -> `{"spoof_score": float}`
- `POST /predict/images` — multipart form data, `files` field with images -> `{"spoof_score": float}`
- Score: 0.0 = real, 1.0 = spoof. Threshold: 0.6

## Key Conventions

- Frontend is a single `demo/index.html` with embedded CSS/JS — no build step
- Backend serves the demo via FastAPI StaticFiles mount at `/demo`
- CORS is enabled on the backend for flexibility
- Save implementation plans to `PLAN.md` in the project root
- Model checkpoint (`checkpoints/resnet50_best.pth`) is gitignored — must be present locally
- Config: `config.yaml` at repo root, overridable via `FAS_CONFIG` env var
