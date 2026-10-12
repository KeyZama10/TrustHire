"""Face-crop extraction for deepfake training / evaluation.

Input: a video manifest CSV with columns `path,label,dataset,split[,method]` (label 1 = fake).
Output: <out>/<dataset>/<split>/<video_id>/<k>.jpg face crops (1.3x box, 256 px) and crops.csv.

Manifest builders for the standard datasets (directory layouts as distributed):
    python -m trusthire.video.prepare manifest --ffpp ROOT --celebdf ROOT --dfdc ROOT --lavdf ROOT -o videos.csv
    python -m trusthire.video.prepare crops videos.csv -o ~/.trusthire/data/faces --frames 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from trusthire.common.face import FaceStack, square_crop
from trusthire.common.media import probe


def manifest_ffpp(root: Path, compression: str = "c23") -> list[dict]:
    """FaceForensics++ (official download script layout). Official train/val/test splits from splits/*.json."""
    rows = []
    split_of: dict[str, str] = {}
    for sp in ("train", "val", "test"):
        f = root / "splits" / f"{sp}.json"
        if f.exists():
            for a, b in json.loads(f.read_text()):
                split_of[a] = split_of[b] = sp
    for p in (root / "original_sequences" / "youtube" / compression / "videos").glob("*.mp4"):
        rows.append(
            {
                "path": str(p),
                "label": 0,
                "dataset": "ffpp",
                "method": "real",
                "split": split_of.get(p.stem, "train"),
            }
        )
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"):
        for p in (root / "manipulated_sequences" / method / compression / "videos").glob("*.mp4"):
            rows.append(
                {
                    "path": str(p),
                    "label": 1,
                    "dataset": "ffpp",
                    "method": method,
                    "split": split_of.get(p.stem.split("_")[0], "train"),
                }
            )
    return rows


def manifest_celebdf(root: Path) -> list[dict]:
    """Celeb-DF v2: List_of_testing_videos.txt defines the official 518-video test set."""
    test = set()
    lst = root / "List_of_testing_videos.txt"
    if lst.exists():
        for line in lst.read_text().splitlines():
            if line.strip():
                test.add(line.split()[1])
    rows = []
    for sub, lab in (("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)):
        for p in (root / sub).glob("*.mp4"):
            rel = f"{sub}/{p.name}"
            rows.append(
                {
                    "path": str(p),
                    "label": lab,
                    "dataset": "celebdf",
                    "method": sub,
                    "split": "test" if rel in test else "train",
                }
            )
    return rows


def manifest_dfdc(root: Path) -> list[dict]:
    """DFDC (Kaggle) part folders with metadata.json."""
    rows = []
    for meta in root.rglob("metadata.json"):
        for name, m in json.loads(meta.read_text()).items():
            rows.append(
                {
                    "path": str(meta.parent / name),
                    "label": int(m["label"] == "FAKE"),
                    "dataset": "dfdc",
                    "method": "dfdc",
                    "split": "test",
                }
            )
    return rows


def manifest_lavdf(root: Path) -> list[dict]:
    """LAV-DF metadata.json (video-level label = any visual modification)."""
    meta = json.loads((root / "metadata.json").read_text())
    return [
        {
            "path": str(root / m["file"]),
            "label": int(bool(m.get("modify_video"))),
            "dataset": "lavdf",
            "method": "lavdf",
            "split": m.get("split", "test"),
        }
        for m in meta
    ]


def extract_crops(manifest: Path, out: Path, frames: int = 32, min_face: int = 64) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    fs = FaceStack.get()
    rows = []
    for _, r in df.iterrows():
        vid = hashlib.md5(r.path.encode()).hexdigest()[:12]
        d = out / r.dataset / r.split / vid
        if (d / "done").exists():
            rows += [
                {**r.to_dict(), "video_id": vid, "crop": str(p), "frame": int(p.stem)}
                for p in sorted(d.glob("*.jpg"))
            ]
            continue
        d.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(r.path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(probe(r.path).duration * 25)
        for k, fi in enumerate(np.linspace(0, max(n - 1, 0), num=frames).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, img = cap.read()
            if not ok:
                continue
            faces = fs.detector.detect(img, max_num=1)
            if not faces or faces[0].size < min_face:
                continue
            p = d / f"{k:03d}.jpg"
            cv2.imwrite(str(p), square_crop(img, faces[0], 1.3, 256), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({**r.to_dict(), "video_id": vid, "crop": str(p), "frame": k})
        cap.release()
        (d / "done").touch()
    crops = pd.DataFrame(rows)
    crops.to_csv(out / "crops.csv", index=False)
    return crops


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    for k in ("ffpp", "celebdf", "dfdc", "lavdf"):
        m.add_argument(f"--{k}", type=Path)
    m.add_argument("-o", type=Path, required=True)
    c = sub.add_parser("crops")
    c.add_argument("manifest", type=Path)
    c.add_argument("-o", type=Path, required=True)
    c.add_argument("--frames", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        rows: list[dict] = []
        for k, fn in (
            ("ffpp", manifest_ffpp),
            ("celebdf", manifest_celebdf),
            ("dfdc", manifest_dfdc),
            ("lavdf", manifest_lavdf),
        ):
            if getattr(a, k):
                rows += fn(getattr(a, k))
        pd.DataFrame(rows).to_csv(a.o, index=False)
        print(f"{len(rows)} videos -> {a.o}")
    else:
        print(len(extract_crops(a.manifest, a.o, a.frames)), "crops")


if __name__ == "__main__":
    main()
