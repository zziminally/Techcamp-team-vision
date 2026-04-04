# Live FAS Demo — Implementation Plan

## Context

Techcamp 2026 project: a live Face Anti-Spoofing demo. Single repo with FastAPI backend (model inference) + vanilla HTML frontend (webcam capture + visual alerts).

## Architecture

- **Backend:** FastAPI in `app/` — serves prediction API + demo frontend via StaticFiles
- **Frontend:** `demo/` — `index.html` + `style.css` + `app.js` (no build step)
- **Model:** ResNet50 checkpoint in `checkpoints/resnet50_best.pth` (180MB, gitignored)
- **Flow:** Webcam -> MediaRecorder (1.5s clips) -> POST `/predict/video` -> parse `spoof_score` -> update UI -> repeat

## What's Done

- [x] FastAPI backend with CORS + StaticFiles mount
- [x] Demo frontend with webcam, spoof detection, camera selector
- [x] Virtual camera detection (OBS) with brightness heuristic
- [x] Separated FE into `index.html` + `style.css` + `app.js`
- [x] Dockerfile for Railway deployment (CPU-only PyTorch, ~1GB image)
- [x] `PORT` env var support for Railway
- [x] Relative API URL (`/predict/video`) — works on any domain

## Deployment

```bash
# Build locally (where the .pth file exists)
docker build -t yourdockerhub/fas-demo:latest .
docker push yourdockerhub/fas-demo:latest

# Railway: New Project → Deploy from Docker Image
```

## Verification

```bash
# Local
uv run python -m app.main        # Backend at :8000
cd demo && python -m http.server 3000  # Frontend at :3000

# Production
# https://<app>.up.railway.app/demo/
# https://<app>.up.railway.app/docs
```
