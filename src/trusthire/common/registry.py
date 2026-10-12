"""Pretrained-weight registry and downloader.

Stdlib-only on purpose: it must run before the heavy environment exists
(`python -m trusthire.common.registry` works with any Python >= 3.10).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


def models_dir() -> Path:
    """Model cache root. Kept outside the repo (and outside OneDrive) by default."""
    root = os.environ.get("TRUSTHIRE_MODELS_DIR") or str(Path.home() / ".trusthire" / "models")
    return Path(root)


def data_dir() -> Path:
    """Large dataset cache (public benchmarks, the custom interview set). Outside the repo by default."""
    root = os.environ.get("TRUSTHIRE_DATA_DIR") or str(Path.home() / ".trusthire" / "data")
    return Path(root)


@dataclass(frozen=True)
class Asset:
    key: str
    url: str
    rel_path: str
    license: str
    source: str
    unzip: bool = False
    profiles: tuple[str, ...] = ("cpu", "gpu")
    note: str = ""
    sha256: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return models_dir() / self.rel_path


GH_DFB = "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1"

ASSETS: list[Asset] = [
    # ---- video deepfake (FF++ c23 trained, DeepfakeBench) ----
    Asset(
        "video.effnb4",
        f"{GH_DFB}/effnb4_best.pth",
        "video/effnb4_best.pth",
        "CC BY-NC 4.0",
        "DeepfakeBench (Yan et al., NeurIPS'23) EfficientNet-B4 trained on FF++ c23",
    ),
    Asset(
        "video.xception",
        f"{GH_DFB}/xception_best.pth",
        "video/xception_best.pth",
        "CC BY-NC 4.0",
        "DeepfakeBench Xception trained on FF++ c23",
    ),
    # ---- voice anti-spoofing ----
    Asset(
        "audio.aasist",
        "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST.pth",
        "audio/AASIST.pth",
        "MIT",
        "clovaai/aasist, trained on ASVspoof 2019 LA",
    ),
    Asset(
        "audio.aasist_l",
        "https://raw.githubusercontent.com/clovaai/aasist/main/models/weights/AASIST-L.pth",
        "audio/AASIST-L.pth",
        "MIT",
        "clovaai/aasist (light), trained on ASVspoof 2019 LA",
    ),
    # ---- AV sync ----
    Asset(
        "avsync.syncnet",
        "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model",
        "avsync/syncnet_v2.model",
        "MIT",
        "joonson/syncnet_python (Chung & Zisserman 2016)",
    ),
    Asset(
        "avsync.example",
        "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi",
        "avsync/example.avi",
        "MIT",
        "SyncNet example talking-head clip (used by tests)",
    ),
    # ---- identity / face detection ----
    Asset(
        "identity.buffalo_l",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "identity/buffalo_l.zip",
        "non-commercial research",
        "InsightFace buffalo_l (SCRFD-10G + ArcFace R50 WebFace600K)",
        unzip=True,
    ),
    Asset(
        "identity.antelopev2",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        "identity/antelopev2.zip",
        "non-commercial research",
        "InsightFace antelopev2 (ArcFace R100 Glint360K)",
        unzip=True,
        profiles=("gpu",),
    ),
    # ---- behaviour ----
    Asset(
        "behavior.face_landmarker",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "behavior/face_landmarker.task",
        "Apache-2.0",
        "MediaPipe Face Landmarker (478 landmarks + 52 blendshapes)",
    ),
    Asset(
        "behavior.yolo26n",
        "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "behavior/yolo26n.pt",
        "AGPL-3.0",
        "Ultralytics YOLO26n (COCO)",
    ),
]

BY_KEY = {a.key: a for a in ASSETS}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "trusthire/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:7.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ensure(key: str, force: bool = False) -> Path:
    """Download asset `key` if missing and return its local path (extracted dir for zips)."""
    a = BY_KEY[key]
    target = a.path.with_suffix("") if a.unzip else a.path
    if target.exists() and not force:
        return target
    print(f"[download] {a.key} <- {a.url}")
    _download(a.url, a.path)
    if a.sha256 and _sha256(a.path) != a.sha256:
        raise RuntimeError(f"checksum mismatch for {a.key}")
    if a.unzip:
        with zipfile.ZipFile(a.path) as z:
            z.extractall(target)
        # some insightface zips nest a same-named folder
        nested = target / target.name
        if nested.is_dir():
            for p in nested.iterdir():
                shutil.move(str(p), target / p.name)
            nested.rmdir()
        a.path.unlink()
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download TrustHire pretrained weights")
    ap.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--only", nargs="*", help="asset keys")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for a in ASSETS:
            print(
                f"{a.key:28} {a.license:24} {'OK ' if a.path.exists() or a.path.with_suffix('').exists() else '-- '} {a.source}"
            )
        return 0
    keys = args.only or [a.key for a in ASSETS if args.profile in a.profiles]
    failed = []
    for k in keys:
        try:
            ensure(k, force=args.force)
        except Exception as e:  # keep going; report at the end
            print(f"[error] {k}: {e}", file=sys.stderr)
            failed.append(k)
    print(f"models dir: {models_dir()}  ({len(keys) - len(failed)}/{len(keys)} ok)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
