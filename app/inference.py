import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from app.config import AppConfig
from app.preprocess import (
    crop_face,
    extract_all_frames_from_video,
    extract_sampled_frames_from_video,
    is_valid_frame,
    preprocess_bgr_image_to_tensor,
    select_torch_device,
)


class NoFaceDetectedError(Exception):
    """Raised when no valid face is detected in any frame."""

    pass


class FASInference:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = select_torch_device(config.model.device)
        self.model = self._load_model()
        self.face_app = (
            self._init_face_detector() if config.face_detection.enabled else None
        )

    def _load_model(self) -> torch.nn.Module:
        arch = self.config.model.arch

        if arch == "resnet50":
            from nets.resnet import resnet50

            model = resnet50(num_classes=self.config.model.num_classes)
        elif arch == "swin_v2_b":
            from nets.swin_transformer_v2 import swin_v2_b

            model = swin_v2_b(num_classes=self.config.model.num_classes, fp16=False)
        elif arch == "mobilenet_v3_small":
            from nets.mobilenetv3 import mobilenet_v3_small

            model = mobilenet_v3_small(num_classes=self.config.model.num_classes)
        elif arch == "shufflenet_v2_x1_0":
            from nets.shufflenetv2 import shufflenet_v2_x1_0

            model = shufflenet_v2_x1_0(num_classes=self.config.model.num_classes)
        elif arch == "shufflenet_v2_x0_5":
            from nets.shufflenetv2 import shufflenet_v2_x0_5

            model = shufflenet_v2_x0_5(num_classes=self.config.model.num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        ckpt_path = Path(self.config.model.checkpoint)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)

            # Remove 'module.' prefix from DDP-trained models
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v

            model.load_state_dict(new_state_dict, strict=False)

        model = model.to(self.device)
        model.eval()
        return model

    def _init_face_detector(self):
        from insightface.app import FaceAnalysis

        providers = [self.config.face_detection.provider, "CPUExecutionProvider"]
        app = FaceAnalysis(providers=providers)
        app.prepare(ctx_id=0, det_size=tuple(self.config.face_detection.det_size))
        return app

    def _extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        Extract evenly spaced valid frames from video.
        Skips black/invalid frames and tries to get max_frames valid ones.
        """
        return extract_sampled_frames_from_video(
            video_path=video_path,
            max_frames=self.config.video.max_frames,
            sampling=self.config.video.sampling,
        )

    @torch.no_grad()
    def predict_frames(self, frames: List[np.ndarray]) -> Tuple[float, List[float]]:
        """
        Predict spoof score from list of frames.
        Returns (average_score, list_of_frame_scores).
        Raises NoFaceDetectedError if no valid faces are found.
        """
        if not frames:
            raise NoFaceDetectedError("No valid frames provided")

        # Crop faces and filter invalid
        cropped_faces = []
        for frame in frames:
            face = crop_face(
                frame_bgr=frame,
                face_app=self.face_app,
                bbox_expand=self.config.face_detection.bbox_expand,
            )
            if face is not None and face.size > 0:
                cropped_faces.append(face)

        if not cropped_faces:
            raise NoFaceDetectedError("No valid face detected in any frame")

        # Preprocess and batch
        tensors = [
            preprocess_bgr_image_to_tensor(face, input_size=self.config.model.input_size)
            for face in cropped_faces
        ]
        batch = torch.stack(tensors).to(self.device)

        # Inference
        outputs = self.model(batch)

        # Handle models returning (features, logits) tuple
        if isinstance(outputs, tuple):
            outputs = outputs[1]  # logits

        probs = torch.softmax(outputs, dim=1)

        # Training label: class 0 = real, class 1 = fake
        # spoof_score = P(fake) = probs[:, 1]
        spoof_scores = probs[:, 1].cpu().numpy()

        avg_score = float(np.mean(spoof_scores))
        frame_scores = [float(s) for s in spoof_scores]

        return avg_score, frame_scores

    def predict_video(self, video_bytes: bytes) -> Tuple[float, List[float]]:
        """Predict spoof score from video bytes."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
            f.write(video_bytes)
            f.flush()
            frames = self._extract_frames_from_video(f.name)

        return self.predict_frames(frames)

    def predict_images(self, images: List[bytes]) -> Tuple[float, List[float]]:
        """Predict spoof score from list of image bytes."""
        frames = []
        for img_bytes in images:
            arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)

        return self.predict_frames(frames)
