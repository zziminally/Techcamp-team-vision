# Face Anti-Spoofing Live Demo

Real-time face anti-spoofing detection for virtual meetings. Detects **replay attacks** (phone screens), **print attacks** (printed photos), and **digital attacks** (OBS virtual camera / deepfakes) — all from your browser.

Built for **Techcamp 2026 — Team Vision**.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Model checkpoint: `checkpoints/resnet50_best.pth`

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the repo
git clone https://github.com/your-org/Techcamp-team-vision.git
cd Techcamp-team-vision

# Install dependencies
uv sync --extra mps    # macOS (Apple Silicon)
uv sync --extra gpu    # Linux/Windows (CUDA)

# Place the model checkpoint
mkdir -p checkpoints
cp /path/to/resnet50_best.pth checkpoints/
```

## Run

```bash
# Start the backend API
uv run python -m app.main

# Start the frontend (separate terminal)
cd demo && python -m http.server 3000
```

- Backend API: **http://localhost:8000/docs**
- Demo page: **http://localhost:3000**

## Demo Flow

1. Click **Start Detection** — your webcam feed appears
2. **Real face** — green border, score near 0.0
3. **Hold a phone screen** (replay attack) — red border, "SPOOF DETECTED"
4. **Switch to OBS Virtual Camera** (digital attack) — red border, "VIRTUAL CAMERA — DIGITAL ATTACK"

Use the **camera dropdown** to switch between your real webcam and OBS Virtual Camera live during the demo.

### Setting Up OBS for Digital Attack Demo

1. Install [OBS Studio](https://obsproject.com) (`brew install --cask obs` on macOS)
2. Add a **Media Source** with a deepfake video, check **Loop**
3. Click **Start Virtual Camera**
4. In the demo page, select "OBS Virtual Camera" from the camera dropdown

## API

The server exposes two prediction endpoints:

| Endpoint | Input | Output |
|---|---|---|
| `POST /predict/video` | Video file (mp4, webm, etc.) | `{"spoof_score": float}` |
| `POST /predict/images` | Image file(s) (jpg, png, etc.) | `{"spoof_score": float}` |

- Score **0.0** = real face
- Score **1.0** = spoof/fake
- Threshold: **0.6**

Full API docs available at **http://localhost:8000/docs**.

## Project Structure

```
├── app/                  # FastAPI backend
│   ├── main.py           # Server + routes
│   ├── inference.py      # Model inference engine
│   ├── config.py         # Configuration loader
│   └── preprocess.py     # Image/video preprocessing
├── nets/                 # Neural network architectures
├── scripts/              # Training & dataset tools
├── demo/                 # Frontend
│   ├── index.html        # Page structure
│   ├── style.css         # Styles
│   └── app.js            # Logic (webcam, API, UI)
├── checkpoints/          # Model weights (gitignored)
├── config.yaml           # Runtime configuration
└── pyproject.toml        # Dependencies
```

## Configuration

Edit `config.yaml` to change model, device, or server settings:

```yaml
model:
  arch: "resnet50"
  device: "cpu"           # "cuda" or "cpu"
  checkpoint: "checkpoints/resnet50_best.pth"

server:
  host: "0.0.0.0"
  port: 8000
```

## License

Techcamp 2026 — Team Vision
