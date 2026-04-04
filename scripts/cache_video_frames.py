from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2

from app.preprocess import is_valid_frame


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_and_save_frames(video_path: Path, out_dir: Path) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if is_valid_frame(frame):
            # zero-padded index for ordering
            out_path = out_dir / f"{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1
    cap.release()
    return {"frames_total": idx, "frames_saved": saved}


def iter_videos(root: Path) -> List[Path]:
    videos: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root (contains videos/)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Cache root dir (default: <data_dir>/video_frames_cache)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Which split to cache (uses data/splits/*.jsonl).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (data_dir / "video_frames_cache")

    split_paths = []
    if args.subset == "train":
        split_paths = [data_dir / "splits" / "train.jsonl"]
    elif args.subset == "test":
        split_paths = [data_dir / "splits" / "test.jsonl"]
    else:
        split_paths = [data_dir / "splits" / "train.jsonl", data_dir / "splits" / "test.jsonl"]

    video_rel_paths = set()
    for sp in split_paths:
        with open(sp, "r", encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                if it.get("type") == "video":
                    video_rel_paths.add(it["path"])

    video_paths = [data_dir / rel for rel in sorted(video_rel_paths)]
    if not video_paths:
        raise RuntimeError("No video items found in selected split(s).")

    manifest = {}
    for i, vp in enumerate(video_paths, start=1):
        rel = vp.relative_to(data_dir)
        cache_dir = out_dir / rel.with_suffix("")  # keep subdirs, drop extension
        done_flag = cache_dir / "_DONE.json"
        if done_flag.exists():
            continue
        if not vp.exists():
            continue
        stats = extract_and_save_frames(vp, cache_dir)
        with open(done_flag, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
        manifest[str(rel)] = stats
        print(f"[{i}/{len(video_paths)}] cached {rel} -> {stats}", flush=True)

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] manifest saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

