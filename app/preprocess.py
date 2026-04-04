from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import torch


def select_torch_device(requested: str) -> torch.device:
    req = (requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but torch.cuda is not available.")
        return torch.device(req)

    if req == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device but torch.backends.mps is not available.")
        return torch.device("mps")

    if req == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown torch device spec: {requested}")


def is_valid_frame(frame_bgr: np.ndarray) -> bool:
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 10.0:
        return False
    if float(np.std(gray)) < 5.0:
        return False
    return True


def crop_face(
    frame_bgr: np.ndarray,
    face_app,
    bbox_expand: int,
) -> Optional[np.ndarray]:
    if not is_valid_frame(frame_bgr):
        return None

    h, w = frame_bgr.shape[:2]
    delta = int(bbox_expand)

    if face_app is not None and h >= 700 and w >= 700:
        faces = face_app.get(frame_bgr)
        if faces:
            bbox = faces[0]["bbox"]
            x1 = max(0, int(bbox[0]) - delta)
            y1 = max(0, int(bbox[1]) - delta)
            x2 = min(w, int(bbox[2]) + delta)
            y2 = min(h, int(bbox[3]) + delta)
            cropped = frame_bgr[y1:y2, x1:x2]
            if cropped.size > 0:
                return cropped

    if h > 500 and w > 500:
        cy, cx = h // 2, w // 2
        return frame_bgr[cy - 250 : cy + 250, cx - 250 : cx + 250]

    return frame_bgr


def preprocess_bgr_image_to_tensor(image_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    size = int(input_size)
    img = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()


def extract_sampled_frames_from_video(
    video_path: str,
    max_frames: int,
    sampling: str = "uniform",
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    n_frames = int(max_frames)
    sample_count = min(total, n_frames * 3)

    sampling = (sampling or "uniform").lower()
    if sampling == "random":
        indices = np.random.randint(0, total, size=sample_count, dtype=np.int64)
        indices = np.unique(indices)
        indices.sort()
    else:
        indices = np.linspace(0, total - 1, sample_count, dtype=np.int64)

    valid_frames: List[np.ndarray] = []
    for idx in indices:
        if len(valid_frames) >= n_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and is_valid_frame(frame):
            valid_frames.append(frame)

    cap.release()
    return valid_frames[:n_frames]


def extract_all_frames_from_video(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if is_valid_frame(frame):
            frames.append(frame)
    cap.release()
    return frames

