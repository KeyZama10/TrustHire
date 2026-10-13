"""Launch / monitor / collect TrustHire GPU jobs on Kaggle.

    python scripts/kaggle_job.py push   audio_reverb_ft      # create or update the kernel and start a GPU run
    python scripts/kaggle_job.py status audio_reverb_ft
    python scripts/kaggle_job.py fetch  audio_reverb_ft      # download outputs -> ~/.trusthire/kaggle/<job>/

Auth: ~/.kaggle/access_token (new KGAT_ tokens) or ~/.kaggle/kaggle.json. TLS uses the Windows/OS certificate
store (truststore) because HTTPS-scanning antivirus breaks certifi-based verification on some machines.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = "import sys, truststore; truststore.inject_into_ssl(); from kaggle.cli import main; sys.argv[0] = 'kaggle'; main()"


def user() -> str:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        return str(json.loads(cfg.read_text())["username"])
    for line in run("kernels", "list", "--mine", "--page-size", "1").splitlines()[2:]:
        return line.split()[0].split("/")[0]
    raise SystemExit("cannot determine Kaggle username")


def slug(job: str) -> str:
    return f"trusthire-{job.replace('_', '-')}"


def run(*args: str) -> str:
    # UTF-8 mode: Kaggle logs contain non-cp1252 characters and the CLI writes them with the default encoding
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run(
        [sys.executable, "-c", CLI, *args], capture_output=True, text=True, encoding="utf-8", env=env
    )
    if out.returncode:
        raise SystemExit(out.stderr or out.stdout)
    return out.stdout


def _lines(p: Path) -> list[str]:
    return (
        [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        if p.exists()
        else []
    )


def push(job: str) -> None:
    src = ROOT / "kaggle" / job
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(src / "train.py", Path(d) / "train.py")
        meta = {
            "id": f"{user()}/{slug(job)}",
            "title": slug(job),
            "code_file": "train.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": _lines(src / "datasets.txt"),
            "competition_sources": [],
            "kernel_sources": [],
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
