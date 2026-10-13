"""Launch / monitor / collect TrustHire GPU jobs on Kaggle.

    python scripts/kaggle_job.py push   audio_reverb_ft      # create or update the kernel and start a GPU run
    python scripts/kaggle_job.py status audio_reverb_ft
    python scripts/kaggle_job.py fetch  audio_reverb_ft      # download outputs -> ~/.trusthire/kaggle/<job>/

Needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")


def user() -> str:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.exists():
        raise SystemExit(f"missing {cfg} — create an API token on kaggle.com > Settings > API")
    return str(json.loads(cfg.read_text())["username"])


def slug(job: str) -> str:
    return f"trusthire-{job.replace('_', '-')}"


def run(*args: str) -> str:
    out = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr or out.stdout)
    return out.stdout


def push(job: str) -> None:
    src = ROOT / "kaggle" / job
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(src / "train.py", Path(d) / "train.py")
        meta = {
            "id": f"{user()}/{slug(job)}", "title": slug(job), "code_file": "train.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        }
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        print(run("kernels", "push", "-p", d))


def status(job: str) -> None:
    print(run("kernels", "status", f"{user()}/{slug(job)}"))


def fetch(job: str) -> None:
    out = Path.home() / ".trusthire" / "kaggle" / job
    out.mkdir(parents=True, exist_ok=True)
    print(run("kernels", "output", f"{user()}/{slug(job)}", "-p", str(out)))
    print("saved to", out)


if __name__ == "__main__":
    cmd, job = sys.argv[1], sys.argv[2]
    {"push": push, "status": status, "fetch": fetch}[cmd](job)
