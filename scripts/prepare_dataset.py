from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def _make_items(data_dir: Path) -> List[Dict]:
    real_videos = data_dir / "videos" / "real"
    fake_videos = data_dir / "videos" / "fake"
    real_images = data_dir / "images" / "real"
    fake_images = data_dir / "images" / "fake"

    items: List[Dict] = []

    for p in _collect_files(real_images):
        if p.suffix.lower() in IMAGE_EXTS:
            items.append({"path": str(p.relative_to(data_dir)), "label": 0, "type": "image"})
    for p in _collect_files(fake_images):
        if p.suffix.lower() in IMAGE_EXTS:
            items.append({"path": str(p.relative_to(data_dir)), "label": 1, "type": "image"})
    for p in _collect_files(real_videos):
        if p.suffix.lower() in VIDEO_EXTS:
            items.append({"path": str(p.relative_to(data_dir)), "label": 0, "type": "video"})
    for p in _collect_files(fake_videos):
        if p.suffix.lower() in VIDEO_EXTS:
            items.append({"path": str(p.relative_to(data_dir)), "label": 1, "type": "video"})

    return items


def _split_items(items: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_train = int(len(items) * train_ratio)
    return items[:n_train], items[n_train:]


def _write_jsonl(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def prepare_splits(
    data_dir: Path,
    out_dir: Optional[Path],
    train_ratio: float,
    seed: int,
) -> Dict:
    out_dir = out_dir or (data_dir / "splits")

    items = _make_items(data_dir)
    if not items:
        raise RuntimeError(f"No dataset files found under: {data_dir}")

    train_items, test_items = _split_items(items, train_ratio=float(train_ratio), seed=int(seed))

    _write_jsonl(out_dir / "train.jsonl", train_items)
    _write_jsonl(out_dir / "test.jsonl", test_items)

    n_img_train = sum(1 for x in train_items if x["type"] == "image")
    n_vid_train = sum(1 for x in train_items if x["type"] == "video")
    n_img_test = sum(1 for x in test_items if x["type"] == "image")
    n_vid_test = sum(1 for x in test_items if x["type"] == "video")
    n_real_train = sum(1 for x in train_items if x["label"] == 0)
    n_fake_train = sum(1 for x in train_items if x["label"] == 1)
    n_real_test = sum(1 for x in test_items if x["label"] == 0)
    n_fake_test = sum(1 for x in test_items if x["label"] == 1)

    summary = {
        "data_dir": str(data_dir),
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "counts": {
            "total": len(items),
            "train": len(train_items),
            "test": len(test_items),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Split Summary]", flush=True)
    print(f"  Train: {len(train_items)} (img={n_img_train}, vid={n_vid_train}, real={n_real_train}, fake={n_fake_train})", flush=True)
    print(f"  Test:  {len(test_items)} (img={n_img_test}, vid={n_vid_test}, real={n_real_test}, fake={n_fake_test})", flush=True)
    print(f"  Output: {out_dir.resolve()}", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=240)
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for splits")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    prepare_splits(
        data_dir=data_dir,
        out_dir=out_dir,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()

