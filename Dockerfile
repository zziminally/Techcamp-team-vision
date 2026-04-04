FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --python .venv/bin/python -e ".[mps]"

# Application code
COPY app/ app/
COPY nets/ nets/
COPY demo/ demo/
COPY config.yaml .

# Model checkpoint (gitignored — must exist locally at build time)
COPY checkpoints/resnet50_best.pth checkpoints/resnet50_best.pth

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["python", "-m", "app.main"]
