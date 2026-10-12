"""faster-whisper ASR with word timestamps (CTranslate2; int8 on CPU, fp16 on GPU)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from trusthire.common.registry import models_dir

# Whisper tends to drop disfluencies; a disfluent prompt makes it transcribe them (well-known trick).
FILLER_PROMPT = "Umm, let me think, like, hmm... Okay, uh, so here's what I, um, think."


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float


@dataclass
class Transcript:
    text: str
    words: list[Word]
    language: str
    avg_logprob: float


class ASR:
    _inst: ClassVar[dict[str, ASR]] = {}
    _lock = threading.Lock()

    def __init__(self, model: str = "small", compute_type: str = "int8"):
        import torch
        from faster_whisper import WhisperModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        self.model = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
            download_root=str(models_dir() / "whisper"),
        )
        self.name = model
        self._run_lock = threading.Lock()

    @classmethod
    def get(cls, model: str = "small", compute_type: str = "int8") -> ASR:
        with cls._lock:
            if model not in cls._inst:
                cls._inst[model] = ASR(model, compute_type)
            return cls._inst[model]

    def transcribe(self, audio: np.ndarray, offset_s: float = 0.0, language: str | None = None) -> Transcript:
        with self._run_lock:
            segs, info = self.model.transcribe(
                audio.astype(np.float32),
                language=language,
                word_timestamps=True,
                vad_filter=True,
                initial_prompt=FILLER_PROMPT,
                condition_on_previous_text=False,
                beam_size=1,
            )
            segs = list(segs)
        words = [
            Word(w.start + offset_s, w.end + offset_s, w.word.strip(), w.probability)
            for s in segs
            for w in (s.words or [])
        ]
        lp = float(np.mean([s.avg_logprob for s in segs])) if segs else float("nan")
        return Transcript(" ".join(s.text.strip() for s in segs).strip(), words, info.language, lp)
