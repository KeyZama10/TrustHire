"""Audio-visual synchronisation per window.

SyncNet backend (default): lip embeddings of 5-frame stacks vs MFCC embeddings, searched over ±vshift
video frames. Outputs the best offset (frames @25 fps, + = audio leads… as in syncnet_python) and the
confidence = median(dist) − min(dist). Dubbed / lip-synced deepfakes give low confidence or large offsets.

AV-HuBERT backend (optional, GPU profile): agreement between audio-only and visual-only AV-HuBERT
encoder features (AVH-Align style). Needs the AV-HuBERT-S2S code — see docs/MODEL_CARDS.md.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import torch

from trusthire.avsync.model import AudioTower, LipTower, build_syncnet, mfcc
from trusthire.common.face import anchor_indices, detect_cached, interpolated_faces, syncnet_crop
from trusthire.common.ort import make_session
from trusthire.common.registry import ensure, models_dir
from trusthire.common.types import Modality, ModuleOutput, Window, missing

FPS = 25


def ensure_onnx() -> tuple[str, str]:
    d = models_dir() / "avsync"
    lip, aud = d / "syncnet_lip.onnx", d / "syncnet_aud.onnx"
    if not (lip.exists() and aud.exists()):
        from trusthire.common.export import export_onnx

        s = build_syncnet(ensure("avsync.syncnet"))
        export_onnx(LipTower(s), (torch.rand(2, 3, 5, 224, 224) * 255,), lip, ["frames"], ["emb"], atol=1e-2)
        export_onnx(AudioTower(s), (torch.randn(2, 1, 13, 20) * 10,), aud, ["mfcc"], ["emb"], atol=1e-2)
    return str(lip), str(aud)


def sync_risk(offset: int, conf: float) -> float:
    """Map (offset, confidence) to risk in [0,1]. Hand-set prior; fusion re-calibrates on labelled data.

    Genuine webcam speech typically gives conf > 3 and |offset| <= 2 frames (syncnet_python reports
    conf ~ 5-10 on clean talking heads); out-of-sync / dubbed audio drops below ~1.5.
    """
    conf_term = 1.0 / (1.0 + math.exp((conf - 2.5) / 0.6))
    # webcam/WebRTC capture adds a constant A/V skew of up to ~120 ms (3 frames) even for genuine speakers
    off_term = float(np.clip((abs(offset) - 3) / 4.0, 0.0, 1.0)) * min(1.0, conf / 3.0)
    return float(1.0 - (1.0 - conf_term) * (1.0 - off_term))


class SyncNetScorer:
    def __init__(self, vshift: int = 7, max_stacks: int = 12):
        lip, aud = ensure_onnx()
        self.lip = make_session(lip)
        self.aud = make_session(aud)
        self.vshift = vshift
        self.max_stacks = max_stacks
        self._lock = threading.Lock()

    def offsets(self, crops_bgr: list[np.ndarray], audio: np.ndarray) -> tuple[int, float, np.ndarray]:
        """crops: consecutive 25 fps 224x224 BGR crops; audio: aligned float32 16 kHz."""
        n = min(len(crops_bgr), len(audio) // 640)
        last = n - 5
        if last < 1:
            raise ValueError("window too short for SyncNet")
        starts = np.unique(np.linspace(0, last - 1, num=min(self.max_stacks, last)).round().astype(int))
        frames = np.stack(crops_bgr[:n]).astype(np.float32)  # (n,224,224,3)
        lip_in = np.stack([frames[v : v + 5].transpose(3, 0, 1, 2) for v in starts])  # (B,3,5,224,224)
        cc = mfcc(audio)  # (13, T)
        # audio features for every frame position that any shifted comparison needs
        positions = range(-self.vshift, last + self.vshift)
        aud_in, valid = [], []
        for v in positions:
            a0 = v * 4
            if a0 < 0 or a0 + 20 > cc.shape[1]:
                aud_in.append(np.zeros((1, 13, 20), np.float32))
                valid.append(False)
            else:
                aud_in.append(cc[None, :, a0 : a0 + 20])
                valid.append(True)
        with self._lock:
            lip_f = self.lip.run(None, {"frames": lip_in})[0]
            aud_f = self.aud.run(None, {"mfcc": np.stack(aud_in)})[0]
        valid_a = np.array(valid)
        dists = np.full((len(starts), 2 * self.vshift + 1), np.nan, np.float32)
        for i, v in enumerate(starts):
            for j, s in enumerate(range(-self.vshift, self.vshift + 1)):
                k = v + s + self.vshift
                if 0 <= k < len(aud_f) and valid_a[k]:
                    dists[i, j] = float(np.linalg.norm(lip_f[i] - aud_f[k]))
        mdist = np.nanmean(dists, axis=0)
        jmin = int(np.nanargmin(mdist))
        offset = self.vshift - jmin  # same sign convention as syncnet_python
        conf = float(np.nanmedian(mdist) - mdist[jmin])
        return offset, conf, mdist


class AVSyncDetector:
    def __init__(self, backend: str = "syncnet", **kw: int):
        if backend != "syncnet":
            raise NotImplementedError("AV-HuBERT backend: see trusthire.avsync.avhubert (GPU profile)")
        self.scorer = SyncNetScorer(**kw)

    def process(self, w: Window, speech_ratio: float | None = None) -> ModuleOutput:
        t0 = time.perf_counter()
        if len(w.frames) < 10:
            return missing(Modality.AVSYNC, w.t_start, w.t_end, "too few frames")
        if speech_ratio is not None and speech_ratio < 0.3:
            return missing(Modality.AVSYNC, w.t_start, w.t_end, "not enough speech")
        # make sure a few anchor detections exist, then interpolate the face track for all frames
        for i in anchor_indices(len(w.frames)):
            detect_cached(w.cache, i, w.frames[i])
        track = interpolated_faces(w.cache, len(w.frames))
        if all(f is None for f in track):
            return missing(Modality.AVSYNC, w.t_start, w.t_end, "no face")
        crops = [
            syncnet_crop(fr, f) if f is not None else np.zeros((224, 224, 3), np.uint8)
            for fr, f in zip(w.frames, track, strict=True)
        ]
        have = sum(1 for k, v in w.cache.items() if k.startswith("faces:") and v)
        n_anchor = sum(1 for k in w.cache if k.startswith("faces:"))
        try:
            offset, conf, mdist = self.scorer.offsets(crops, w.audio)
        except ValueError as e:
            return missing(Modality.AVSYNC, w.t_start, w.t_end, str(e))
        risk = sync_risk(offset, conf)
        face_cov = have / max(1, n_anchor)
        module_conf = float(
            np.clip(face_cov * (speech_ratio if speech_ratio is not None else 1.0) / 0.6, 0, 1)
        )
        return ModuleOutput(
            module=Modality.AVSYNC,
            score=risk,
            confidence=module_conf,
            t_start=w.t_start,
            t_end=w.t_end,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            details={
                "offset_frames": offset,
                "offset_ms": offset * 40,
                "sync_conf": round(conf, 3),
                "min_dist": round(float(np.nanmin(mdist)), 3),
                "backend": "syncnet",
            },
        )
