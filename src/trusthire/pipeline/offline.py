"""Offline analysis of a recorded interview (used by the Celery worker, the CLI and the eval harness)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trusthire.common.media import SR, iter_frames, load_audio, probe
from trusthire.common.settings import get_settings
from trusthire.common.types import FusedPoint
from trusthire.fusion.live import session_trust
from trusthire.identity.infer import Enrollment
from trusthire.pipeline.analyzer import AnalyzerConfig, SessionAnalyzer
from trusthire.pipeline.chunker import SlidingChunker


@dataclass
class OfflineResult:
    points: list[FusedPoint]
    answers: list[dict]
    trust: dict[str, float]
    latency_ms: list[float]
    duration_s: float
    wall_s: float
    flagged_frames: list[tuple[float, np.ndarray, float]] = field(
        default_factory=list
    )  # (t, face crop, prob)


def analyze_file(
    path: str | Path,
    enrollment: Enrollment | None = None,
    resume_text: str | None = None,
    config: AnalyzerConfig | None = None,
    progress: Callable[[float], None] | None = None,
    keep_flagged: int = 6,
) -> OfflineResult:
    s = get_settings()
    cfg = config or AnalyzerConfig()
    cfg.synchronous_transcript = False
    info = probe(path)
    audio = load_audio(path)
    an = SessionAnalyzer(enrollment, resume_text, config=cfg, settings=s)
    ch = SlidingChunker(s.window_s, s.hop_s)
    pushed = 0
    flagged: list[tuple[float, np.ndarray, float]] = []
    t_wall = time.perf_counter()

    def run(windows: list) -> None:
        for w in windows:
            p = an.process_window(w)
            for idx, crop, prob in w.cache.get("video_crops", []):
                flagged.append((w.frame_times[idx], crop, float(prob)))
            flagged.sort(key=lambda x: -x[2])
            del flagged[keep_flagged * 3 :]
            if progress and info.duration:
                progress(min(0.99, p.t_end / info.duration))

    try:
        for t, frame in iter_frames(path, s.analysis_fps):
            ch.push_frame(t, frame)
            target = int(min(len(audio), (t + 1 / s.analysis_fps) * SR))
            if target > pushed:
                ch.push_audio(audio[pushed:target])
                pushed = target
            run(ch.pop_windows())
        if pushed < len(audio):
            ch.push_audio(audio[pushed:])
        run(ch.pop_windows(flush=True))
        an.finish()
        answers = [
            {
                "t_start": r.t_start,
                "t_end": r.t_end,
                "text": r.text,
                "features": r.features,
                "ai_text_prob": r.ai_text_prob,
                "scripted_score": r.scripted.score,
                "resume": None if r.resume is None else {"score": r.resume.score, **r.resume.details},
                "words": r.words,
            }
            for r in (an.transcript.results if an.transcript else [])
        ]
        pts = list(an.points)
        # de-duplicate flagged frames that are < 1 s apart, keep the most suspicious
        keep: list[tuple[float, np.ndarray, float]] = []
        for f in sorted(flagged, key=lambda x: -x[2]):
            if all(abs(f[0] - k[0]) > 1.0 for k in keep):
                keep.append(f)
            if len(keep) >= keep_flagged:
                break
        return OfflineResult(
            pts,
            answers,
            session_trust(pts),
            list(an.latencies),
            info.duration,
            time.perf_counter() - t_wall,
            keep,
        )
    finally:
        an.close()
