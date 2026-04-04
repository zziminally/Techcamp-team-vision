from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class ModelConfig:
    arch: str = "resnet50"
    num_classes: int = 2
    checkpoint: str = "checkpoints/resnet50_best.pth"
    input_size: int = 224
    device: str = "cuda"


@dataclass
class FaceDetectionConfig:
    enabled: bool = True
    provider: str = "CUDAExecutionProvider"
    det_size: List[int] = field(default_factory=lambda: [640, 640])
    bbox_expand: int = 20


@dataclass
class VideoConfig:
    max_frames: int = 5
    sampling: str = "uniform"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    face_detection: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        return AppConfig()

    with open(path) as f:
        data = yaml.safe_load(f)

    return AppConfig(
        model=ModelConfig(**data.get("model", {})),
        face_detection=FaceDetectionConfig(**data.get("face_detection", {})),
        video=VideoConfig(**data.get("video", {})),
        server=ServerConfig(**data.get("server", {})),
    )
