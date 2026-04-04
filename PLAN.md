# Live FAS Demo Page — Implementation Plan

## Context

Techcamp 2026 project: a live Face Anti-Spoofing demo. Single repo with FastAPI backend (model inference) + vanilla HTML frontend (webcam capture + visual alerts).

## Architecture

- **Backend:** FastAPI in `app/` — serves prediction API + demo frontend via StaticFiles
- **Frontend:** Single `demo/index.html` with embedded CSS/JS — no build step
- **Model:** ResNet50 checkpoint in `checkpoints/resnet50_best.pth` (180MB, gitignored)
- **Flow:** Webcam -> MediaRecorder (1.5s clips) -> POST `/predict/video` -> parse `spoof_score` -> update UI -> repeat

## What's Done

- [x] Added CORS middleware to `app/main.py`
- [x] Added StaticFiles mount for `demo/` at `/demo`
- [x] Updated `pyproject.toml` project name
- [x] Updated `.gitignore` for consolidated repo

## What's Left

- [ ] Create `demo/index.html` — the frontend demo page
  - Webcam access via `getUserMedia`
  - MediaRecorder for 1.5s video clips (WebM on Chrome/Firefox, MP4 on Safari)
  - POST clips to `/predict/video`, parse `{"spoof_score": float}`
  - Threshold 0.6: green border + "Real Face" vs red border + warning banner + "SPOOF DETECTED"
  - Error handling: camera denied, API down, no face detected
  - Score bar visualization

## Verification

```bash
uv run python -m app.main
# Open http://localhost:8000/demo/
```

1. Click "Start Detection" -> webcam activates, clips sent, score updates
2. Real face -> score near 0, green border
3. Phone screen replay -> score near 1.0, red border + warning banner
4. No face in frame -> "No face detected" message
5. API stopped -> error message, resumes when API restarts
