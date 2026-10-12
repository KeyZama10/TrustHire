"""Per-chunk latency + memory benchmark of the full analyzer (transcript excluded: it is asynchronous).

    python -m trusthire.evaluation.latency [--video path] [--windows N]
Writes results/latency/latency_<profile>.json and a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import psutil

from trusthire.common.media import iter_frames, load_audio
from trusthire.common.registry import ensure
from trusthire.common.settings import REPO_ROOT, get_settings
from trusthire.pipeline.analyzer import AnalyzerConfig, SessionAnalyzer
from trusthire.pipeline.chunker import SlidingChunker


def run(video: str | None = None, max_windows: int = 30, repeats: int = 2) -> dict:
    s = get_settings()
    path = video or str(ensure("avsync.example"))
    frames = list(iter_frames(path, s.analysis_fps))
    audio = load_audio(path)
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    an = SessionAnalyzer(config=AnalyzerConfig(transcript=False), settings=s)
    per_mod: dict[str, list[float]] = {}
    totals: list[float] = []
    peak = rss0
    try:
        for rep in range(repeats):  # repeat the clip to collect enough windows
            ch = SlidingChunker(s.window_s, s.hop_s)
            off = rep * (len(frames) / s.analysis_fps + 10)
            for t, f in frames:
                ch.push_frame(t + off, f)
            ch._audio_t0 = off
            ch._next_start = off
            ch.push_audio(audio)
            for w in ch.pop_windows():
                t0 = time.perf_counter()
                p = an.process_window(w)
                totals.append((time.perf_counter() - t0) * 1e3)
                for k, o in p.modules.items():
                    if not o.details.get("carried") and not o.details.get("missing") and o.latency_ms > 0:
                        per_mod.setdefault(k, []).append(o.latency_ms)
                peak = max(peak, proc.memory_info().rss)
                if len(totals) >= max_windows:
                    break
    finally:
        an.close()
    warm = totals[2:] if len(totals) > 4 else totals  # skip warm-up windows
    res = {
        "profile": s.profile,
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "on_battery": bool(psutil.sensors_battery() and not psutil.sensors_battery().power_plugged),
        "window_s": s.window_s,
        "hop_s": s.hop_s,
        "n_windows": len(warm),
        "total_ms_p50": float(np.median(warm)),
        "total_ms_p95": float(np.percentile(warm, 95)),
        "real_time_factor": float(np.median(warm) / (s.hop_s * 1e3)),
        "modules_ms_p50": {k: float(np.median(v[2:] or v)) for k, v in per_mod.items()},
        "peak_rss_mb": round((peak) / 2**20, 1),
        "gpu_mem_mb": _gpu_mem(),
        "models": {"video": s.video_backbone, "audio": s.audio_model},
    }
    out = REPO_ROOT / "results" / "latency"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"latency_{s.profile}.json").write_text(json.dumps(res, indent=2))
    lines = ["| Module | p50 ms (runs concurrently) |", "|---|---|"]
    lines += [f"| {k} | {v:.0f} |" for k, v in sorted(res["modules_ms_p50"].items(), key=lambda kv: -kv[1])]
    lines += [
        f"| **window total (wall)** | **{res['total_ms_p50']:.0f}** (p95 {res['total_ms_p95']:.0f}) |",
        f"| hop budget | {s.hop_s * 1e3:.0f} |",
    ]
    (out / f"latency_{s.profile}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=2))
    return res


def _gpu_mem() -> float | None:
    try:
        import torch

        return round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--windows", type=int, default=30)
    a = ap.parse_args(argv)
    run(a.video, a.windows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
