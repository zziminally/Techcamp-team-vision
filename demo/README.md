# Demo Frontend

Live face anti-spoofing demo page — vanilla HTML/CSS/JS, no build step.

## Run

```bash
cd demo
python -m http.server 3000
```

Then open **http://localhost:3000** in your browser.

The backend API must be running at **http://localhost:8000** (see root README for backend setup).

## Files

- `index.html` — the entire demo (HTML + CSS + JS, all-in-one)

## How It Works

1. Webcam access via `getUserMedia`
2. Records 1.5s video clips via `MediaRecorder`
3. Sends clips to `POST http://localhost:8000/predict/video`
4. Parses `{"spoof_score": float}` response and updates the UI
5. Virtual camera detection (OBS) via device label heuristic + frame brightness check

## UI States

| State | Border | Trigger |
|---|---|---|
| Safe | Green | Real face, score < 0.6 |
| Spoof | Red | Replay/print attack, score >= 0.6 |
| Virtual Camera | Red | OBS or virtual camera with non-black frame |
