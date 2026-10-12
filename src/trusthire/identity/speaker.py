"""ECAPA-TDNN speaker embeddings (SpeechBrain, VoxCeleb-trained, EER 0.8% on VoxCeleb1-O)."""

from __future__ import annotations

import threading

import numpy as np
import torch

from trusthire.common.registry import models_dir


class SpeakerEncoder:
    _inst: SpeakerEncoder | None = None
    _cls_lock = threading.Lock()

    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb"):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.model = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(models_dir() / "identity" / "ecapa"),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            local_strategy=LocalStrategy.COPY,  # Windows: no symlinks without developer mode
        )
        self.model.eval()
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> SpeakerEncoder:
        with cls._cls_lock:
            if cls._inst is None:
                cls._inst = SpeakerEncoder()
            return cls._inst

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._lock, torch.inference_mode():
            e = (
                self.model.encode_batch(torch.from_numpy(audio.astype(np.float32))[None])
                .reshape(-1)
                .cpu()
                .numpy()
            )
        return e / max(float(np.linalg.norm(e)), 1e-8)
