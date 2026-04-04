from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.preprocess import (
    crop_face,
    extract_all_frames_from_video,
    extract_sampled_frames_from_video,
    preprocess_bgr_image_to_tensor,
    select_torch_device,
)


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(arch: str, num_classes: int) -> torch.nn.Module:
    if arch == "resnet50":
        from nets.resnet import resnet50

        return resnet50(num_classes=num_classes)
    if arch == "swin_v2_b":
        from nets.swin_transformer_v2 import swin_v2_b

        return swin_v2_b(num_classes=num_classes, fp16=False)
    if arch == "mobilenet_v3_small":
        from nets.mobilenetv3 import mobilenet_v3_small

        return mobilenet_v3_small(num_classes=num_classes)
    if arch == "shufflenet_v2_x1_0":
        from nets.shufflenetv2 import shufflenet_v2_x1_0

        return shufflenet_v2_x1_0(num_classes=num_classes)
    if arch == "shufflenet_v2_x0_5":
        from nets.shufflenetv2 import shufflenet_v2_x0_5

        return shufflenet_v2_x0_5(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch}")


def read_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class MixedFASDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        items: List[Dict],
        input_size: int,
        train: bool,
        frames_per_video: int,
        use_face_detection: bool,
        face_app,
        bbox_expand: int,
        video_frames_cache_dir: Path | None = None,
    ):
        self.data_dir = data_dir
        self.items = items
        self.input_size = int(input_size)
        self.train = bool(train)
        self.frames_per_video = int(frames_per_video)
        self.use_face_detection = bool(use_face_detection)
        self.face_app = face_app
        self.bbox_expand = int(bbox_expand)
        self.video_frames_cache_dir = video_frames_cache_dir

    def __len__(self) -> int:
        return len(self.items)

    def _read_image(self, rel_path: str) -> np.ndarray:
        p = self.data_dir / rel_path
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        return img

    def _frames_from_video(self, rel_path: str) -> List[np.ndarray]:
        p = self.data_dir / rel_path

        # If cached frames exist, use them to avoid re-decoding the video every epoch.
        if self.video_frames_cache_dir is not None:
            cache_dir = self.video_frames_cache_dir / Path(rel_path).with_suffix("")
            if cache_dir.exists():
                frame_paths = sorted(
                    [x for x in cache_dir.glob("*.jpg") if x.is_file()]
                    + [x for x in cache_dir.glob("*.png") if x.is_file()]
                )
                if frame_paths:
                    frames: List[np.ndarray] = []
                    if self.train:
                        # sample from cached frames
                        step = max(1, len(frame_paths) // max(1, self.frames_per_video))
                        sampled = frame_paths[::step][: self.frames_per_video]
                        for fp in sampled:
                            img = cv2.imread(str(fp))
                            if img is not None:
                                frames.append(img)
                        return frames
                    else:
                        for fp in frame_paths:
                            img = cv2.imread(str(fp))
                            if img is not None:
                                frames.append(img)
                        return frames

        if self.train:
            return extract_sampled_frames_from_video(
                video_path=str(p),
                max_frames=self.frames_per_video,
                sampling="random",
            )
        return extract_all_frames_from_video(str(p))

    def _maybe_crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.use_face_detection:
            return frame_bgr
        cropped = crop_face(
            frame_bgr=frame_bgr,
            face_app=self.face_app,
            bbox_expand=self.bbox_expand,
        )
        if cropped is None:
            raise RuntimeError("No valid face detected in frame")
        return cropped

    def __getitem__(self, idx: int):
        max_retries = 10
        original_idx = idx
        for attempt in range(max_retries):
            try:
                it = self.items[idx]
                rel_path = it["path"]
                label = int(it["label"])
                typ = it["type"]

                if typ == "image":
                    img = self._read_image(rel_path)
                    img = self._maybe_crop(img)
                    x = preprocess_bgr_image_to_tensor(img, input_size=self.input_size)
                    return x, label, typ

                if typ == "video":
                    frames = self._frames_from_video(rel_path)
                    if not frames:
                        raise RuntimeError(f"No frames extracted from video: {rel_path}")
                    xs = []
                    for fr in frames:
                        try:
                            fr = self._maybe_crop(fr)
                            xs.append(preprocess_bgr_image_to_tensor(fr, input_size=self.input_size))
                        except Exception:
                            continue
                    if not xs:
                        raise RuntimeError(f"No valid frames in video: {rel_path}")
                    x = torch.stack(xs, dim=0)
                    return x, label, typ

                raise ValueError(f"Unknown sample type: {typ}")
            except Exception as e:
                if attempt == 0:
                    print(f"[WARN] Skipping broken sample {rel_path}: {e}", flush=True)
                idx = (idx + 1) % len(self.items)
                if idx == original_idx:
                    break
        raise RuntimeError(f"Failed to load sample after {max_retries} retries")


def collate_mixed(batch):
    images: List[torch.Tensor] = []
    labels: List[int] = []
    types: List[str] = []

    for x, y, t in batch:
        if t == "image":
            images.append(x.unsqueeze(0))
            labels.append(y)
            types.append("image")
        else:
            images.append(x)
            labels.append(y)
            types.append("video")

    return images, torch.tensor(labels, dtype=torch.long), types


@torch.no_grad()
def forward_scores(
    model: torch.nn.Module,
    images: List[torch.Tensor],
    device: torch.device,
) -> List[float]:
    scores: List[float] = []
    for x in images:
        x = x.to(device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")

        outputs = model(x)
        if isinstance(outputs, tuple):
            outputs = outputs[1]
        probs = torch.softmax(outputs, dim=1)
        fake = probs[:, 1].detach().cpu().numpy().astype(np.float64)
        score = float(np.mean(fake))
        scores.append(score)
    return scores


def find_best_threshold(scores: List[float], labels: List[int]) -> Tuple[float, Dict[str, float]]:
    scores_np = np.asarray(scores, dtype=np.float64)
    labels_np = np.asarray(labels, dtype=np.int64)
    uniq = np.unique(scores_np)
    if uniq.size == 1:
        th = float(uniq[0])
        pred = (scores_np >= th).astype(np.int64)
        acc = float((pred == labels_np).mean())
        return th, {"accuracy": acc}

    candidates = (uniq[:-1] + uniq[1:]) / 2.0
    best_th = float(candidates[0])
    best_acc = -1.0
    for th in candidates:
        pred = (scores_np >= th).astype(np.int64)
        acc = float((pred == labels_np).mean())
        if acc > best_acc:
            best_acc = acc
            best_th = float(th)
    return best_th, {"accuracy": float(best_acc)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_types: List[str] = []

    for images, labels, types in loader:
        scores = forward_scores(model, images, device)
        all_scores.extend(scores)
        all_labels.extend(labels.numpy().tolist())
        all_types.extend(types)

    th, th_metrics = find_best_threshold(all_scores, all_labels)
    pred = (np.asarray(all_scores) >= th).astype(np.int64)
    labels_np = np.asarray(all_labels, dtype=np.int64)
    acc = float((pred == labels_np).mean())

    img_mask = np.asarray([t == "image" for t in all_types], dtype=bool)
    vid_mask = ~img_mask

    out: Dict[str, float] = {
        "threshold": float(th),
        "accuracy": acc,
        "threshold_accuracy": float(th_metrics["accuracy"]),
        "n": float(len(all_scores)),
    }

    if img_mask.any():
        acc_img = float((pred[img_mask] == labels_np[img_mask]).mean())
        out["accuracy_image"] = acc_img
        out["n_image"] = float(img_mask.sum())
    if vid_mask.any():
        acc_vid = float((pred[vid_mask] == labels_np[vid_mask]).mean())
        out["accuracy_video"] = acc_vid
        out["n_video"] = float(vid_mask.sum())

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--test_split", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--frames_per_video", type=int, default=16)
    parser.add_argument("--use_face_detection", action="store_true")
    parser.add_argument("--bbox_expand", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument(
        "--video_frames_cache_dir",
        type=str,
        default="",
        help="Optional cache dir containing pre-extracted frames for videos.",
    )
    args = parser.parse_args()

    device = select_torch_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = read_jsonl(Path(args.train_split))
    test_items = read_jsonl(Path(args.test_split))

    video_cache_dir = None
    if args.video_frames_cache_dir:
        video_cache_dir = Path(args.video_frames_cache_dir)

    face_app = None
    if args.use_face_detection:
        from insightface.app import FaceAnalysis

        face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
        face_app.prepare(ctx_id=0, det_size=(640, 640))

    train_ds = MixedFASDataset(
        data_dir=data_dir,
        items=train_items,
        input_size=args.input_size,
        train=True,
        frames_per_video=args.frames_per_video,
        use_face_detection=args.use_face_detection,
        face_app=face_app,
        bbox_expand=args.bbox_expand,
        video_frames_cache_dir=video_cache_dir,
    )
    test_ds = MixedFASDataset(
        data_dir=data_dir,
        items=test_items,
        input_size=args.input_size,
        train=False,
        frames_per_video=args.frames_per_video,
        use_face_detection=args.use_face_detection,
        face_app=face_app,
        bbox_expand=args.bbox_expand,
        video_frames_cache_dir=video_cache_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_mixed,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        collate_fn=collate_mixed,
        pin_memory=False,
    )

    model = load_model(args.arch, num_classes=args.num_classes).to(device)

    if args.pretrain:
        print(f"[INFO] Loading pretrained weights: {args.pretrain}", flush=True)
        ckpt = torch.load(args.pretrain, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"[INFO] Loaded pretrained weights (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_acc = -1.0
    best_path = None

    print(f"[INFO] Starting training: {len(train_loader)} batches per epoch", flush=True)
    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        n_steps = 0
        print(f"[Epoch {epoch+1}/{args.epochs}] Training...", flush=True)

        for batch_idx, (images, labels, types) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"  batch {batch_idx+1}/{len(train_loader)}", flush=True)
            optimizer.zero_grad(set_to_none=True)

            batch_x = []
            batch_y = []
            for x, y, t in zip(images, labels.tolist(), types):
                if t == "image":
                    batch_x.append(x)
                    batch_y.append(y)
                else:
                    for fr in x:
                        batch_x.append(fr.unsqueeze(0))
                        batch_y.append(y)

            x = torch.cat(batch_x, dim=0).to(device)
            y = torch.tensor(batch_y, dtype=torch.long, device=device)

            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu().item())
            n_steps += 1

        metrics = evaluate(model, test_loader, device)
        avg_loss = running_loss / max(1, n_steps)
        acc = float(metrics["accuracy"])
        th = float(metrics["threshold"])

        ckpt = {
            "state_dict": model.state_dict(),
            "arch": args.arch,
            "input_size": int(args.input_size),
            "num_classes": int(args.num_classes),
            "epoch": int(epoch),
            "metrics": metrics,
        }
        last_path = output_dir / "last.pth"
        torch.save(ckpt, last_path)

        if acc > best_acc:
            best_acc = acc
            best_path = output_dir / "best.pth"
            torch.save(ckpt, best_path)

        print(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "loss": avg_loss,
                    "accuracy": acc,
                    "threshold": th,
                },
                ensure_ascii=False,
            )
        )

    if best_path is not None:
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(best.get("metrics", {}), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

