"""Media I/O: decode video+audio from files (PyAV), resampling helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def to_mono_16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        # (n, ch) from soundfile or (ch, n) from PyAV — average the channel axis (the short one)
        x = x.mean(axis=0 if x.shape[0] < x.shape[1] else 1)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def load_audio(path: str | Path) -> np.ndarray:
    """Load any audio (or the audio track of a video) as float32 mono 16 kHz."""
    p = Path(path)
    if p.suffix.lower() in {".wav", ".flac", ".ogg"}:
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        return to_mono_16k(x, sr)
    return decode_audio_track(p)


def decode_audio_track(path: str | Path) -> np.ndarray:
    with av.open(str(path)) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        res = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(audio=0):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


@dataclass
class VideoInfo:
    fps: float
    duration: float
    width: int
    height: int
    has_audio: bool


def probe(path: str | Path) -> VideoInfo:
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        fps = float(vs.average_rate or vs.guessed_rate or 25)
        dur = float(c.duration / av.time_base) if c.duration else float(vs.frames / fps if vs.frames else 0)
        return VideoInfo(fps, dur, vs.codec_context.width, vs.codec_context.height, bool(c.streams.audio))


def iter_frames(
    path: str | Path, target_fps: float = 25.0, max_side: int = 960
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, BGR frame) resampled to `target_fps` (nearest-frame, constant rate)."""
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        next_t = 0.0
        step = 1.0 / target_fps
        last: tuple[float, np.ndarray] | None = None
        for frame in c.decode(video=0):
            t = float(frame.time) if frame.time is not None else (last[0] + step if last else 0.0)
            if t + 1e-6 < next_t:
                continue
            img: np.ndarray = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            if max(h, w) > max_side:
                s = max_side / max(h, w)
                import cv2

                img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            # duplicate frames if the source fps is below target (keeps 25 fps for SyncNet)
            while next_t <= t + 1e-6:
                yield next_t, img
                next_t += step
            last = (t, img)
