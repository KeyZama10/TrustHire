"""Sliding-window chunker: 2-3 s windows with 50 % overlap over synchronized A/V streams.

Works for both live streams (push frames/audio as they arrive) and offline files.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from trusthire.common.types import Window


class SlidingChunker:
    def __init__(self, window_s: float = 3.0, hop_s: float = 1.5, sample_rate: int = 16000):
        if not 0 < hop_s <= window_s:
            raise ValueError("need 0 < hop_s <= window_s")
        self.window_s = window_s
        self.hop_s = hop_s
        self.sr = sample_rate
        self._frames: deque[tuple[float, np.ndarray]] = deque()
        self._audio = np.zeros(0, np.float32)
        self._audio_t0 = 0.0  # stream time of self._audio[0]
        self._next_start = 0.0
        self._index = 0
        self._video_t = -1.0
        self._audio_t = 0.0

    # ---- ingest -------------------------------------------------------
    def push_frame(self, t: float, frame: np.ndarray) -> None:
        self._frames.append((t, frame))
        self._video_t = max(self._video_t, t)

    def push_audio(self, samples: np.ndarray) -> None:
        self._audio = np.concatenate([self._audio, samples.astype(np.float32, copy=False)])
        self._audio_t = self._audio_t0 + len(self._audio) / self.sr

    # ---- emit ---------------------------------------------------------
    def _ready(self, flush: bool) -> bool:
        end = self._next_start + self.window_s
        have_v = self._video_t >= end - 0.05 or self._video_t < 0  # audio-only streams
        have_a = self._audio_t >= end - 1e-6
        if flush:
            return self._next_start < max(self._video_t, self._audio_t) - 0.5
        return have_v and have_a

    def pop_windows(self, flush: bool = False) -> list[Window]:
        out: list[Window] = []
        while self._ready(flush):
            t0, t1 = self._next_start, self._next_start + self.window_s
            frames = [(t, f) for t, f in self._frames if t0 <= t < t1]
            a0 = round((t0 - self._audio_t0) * self.sr)
            a1 = round((t1 - self._audio_t0) * self.sr)
            audio = self._audio[max(a0, 0) : max(a1, 0)]
            if len(audio) < a1 - a0:
                audio = np.pad(audio, (0, a1 - a0 - len(audio)))
            out.append(
                Window(
                    index=self._index,
                    t_start=t0,
                    t_end=t1,
                    frames=[f for _, f in frames],
                    frame_times=[t for t, _ in frames],
                    audio=audio,
                    sample_rate=self.sr,
                )
            )
            self._index += 1
            self._next_start += self.hop_s
            self._trim()
        return out

    def _trim(self) -> None:
        keep_from = self._next_start
        while self._frames and self._frames[0][0] < keep_from:
            self._frames.popleft()
        drop = int((keep_from - self._audio_t0) * self.sr)
        if drop > 0:
            self._audio = self._audio[drop:]
            self._audio_t0 += drop / self.sr
