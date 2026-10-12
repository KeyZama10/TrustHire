"""Identity continuity (proxy detection).

Face: ArcFace (InsightFace) embedding of the candidate's face, re-verified each window against
  (a) the enrolled ID photo / enrollment capture and (b) the session's running identity (EMA),
  so both "wrong person from the start" and "person swapped mid-interview" are caught.
Voice: ECAPA-TDNN embedding of the window's speech against the enrollment voiceprint, same logic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from trusthire.common.face import FaceStack, anchor_indices, detect_cached
from trusthire.common.types import Modality, ModuleOutput, Window, missing


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Enrollment:
    face: np.ndarray | None = None  # unit 512-d ArcFace
    voice: np.ndarray | None = None  # unit 192-d ECAPA
    face_source: str = ""
    voice_source: str = ""
    auto_face: list[np.ndarray] = field(default_factory=list)
    auto_voice: list[np.ndarray] = field(default_factory=list)


def enroll_face_from_image(img_bgr: np.ndarray) -> np.ndarray:
    fs = FaceStack.get()
    faces = fs.detector.detect(img_bgr, max_num=1)
    if not faces:
        raise ValueError("no face found in the enrollment image")
    return fs.arcface.embed(img_bgr, faces[0])


def enroll_voice_from_audio(audio: np.ndarray, min_speech_s: float = 3.0) -> np.ndarray:
    from trusthire.audio.infer import speech_ratio
    from trusthire.identity.speaker import SpeakerEncoder

    if len(audio) / 16000 * speech_ratio(audio) < min_speech_s:
        raise ValueError(f"need at least {min_speech_s:.0f} s of speech for a voiceprint")
    return SpeakerEncoder.get().embed(audio)


class _Track:
    """Running identity embedding with change detection."""

    def __init__(self, alpha: float = 0.15):
        self.ema: np.ndarray | None = None
        self.alpha = alpha
        self.history: list[float] = []

    def update(self, e: np.ndarray) -> float | None:
        prev = None if self.ema is None else float(self.ema @ e)
        if self.ema is None:
            self.ema = e.copy()
        elif prev is not None and prev > 0.2:  # don't absorb a different person into the reference
            self.ema = (1 - self.alpha) * self.ema + self.alpha * e
            self.ema /= np.linalg.norm(self.ema)
        if prev is not None:
            self.history.append(prev)
        return prev


class IdentityMonitor:
    AUTO_ENROLL_WINDOWS = 4  # without an ID photo, the first ~6 s become the reference

    def __init__(self, enrollment: Enrollment, face_thr: float = 0.30, voice_thr: float = 0.25):
        self.enr = enrollment
        self.face_thr = face_thr
        self.voice_thr = voice_thr
        self.face_track = _Track()
        self.voice_track = _Track()

    # ------------------------------------------------------------------ face
    def face(self, w: Window) -> ModuleOutput:
        t0 = time.perf_counter()
        if not w.frames:
            return missing(Modality.FACE_ID, w.t_start, w.t_end, "no video")
        fs = FaceStack.get()
        embs: list[np.ndarray] = []
        n_faces: list[int] = []
        anchors = anchor_indices(len(w.frames))
        # middle anchor first; fall back to the ends only if no usable face there
        for i in [anchors[len(anchors) // 2], anchors[0], anchors[-1]]:
            if embs:
                break
            faces = detect_cached(w.cache, i, w.frames[i])
            n_faces.append(len(faces))
            if faces and faces[0].size >= 48:
                embs.append(fs.arcface.embed(w.frames[i], faces[0]))
        if not embs:
            return missing(Modality.FACE_ID, w.t_start, w.t_end, "no usable face")
        e = np.mean(embs, 0)
        e /= np.linalg.norm(e)
        ref = self.enr.face
        if ref is None:
            self.enr.auto_face.append(e)
            if len(self.enr.auto_face) >= self.AUTO_ENROLL_WINDOWS:
                r = np.mean(self.enr.auto_face, 0)
                self.enr.face = r / np.linalg.norm(r)
                self.enr.face_source = "auto (first seconds of session)"
        sim_enr = float(ref @ e) if ref is not None else None
        sim_prev = self.face_track.update(e)
        # risk: low similarity to enrollment, or a sudden drop vs the running identity
        r_enr = _sigmoid((self.face_thr - sim_enr) / 0.06) if sim_enr is not None else 0.0
        r_jump = _sigmoid((0.35 - sim_prev) / 0.06) if sim_prev is not None else 0.0
        risk = max(r_enr, r_jump)
        conf = 0.9 if sim_enr is not None else (0.4 if sim_prev is not None else 0.1)
        return ModuleOutput(
            module=Modality.FACE_ID,
            score=risk,
            confidence=conf,
            t_start=w.t_start,
            t_end=w.t_end,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            details={
                "sim_enrolled": None if sim_enr is None else round(sim_enr, 3),
                "sim_running": None if sim_prev is None else round(sim_prev, 3),
                "max_faces": max(n_faces),
                "enrollment": self.enr.face_source or "pending",
            },
        )

    # ------------------------------------------------------------------ voice
    def voice(self, w: Window, speech_ratio: float, ctx_audio: np.ndarray | None = None) -> ModuleOutput:
        from trusthire.identity.speaker import SpeakerEncoder

        t0 = time.perf_counter()
        audio = ctx_audio if ctx_audio is not None and len(ctx_audio) >= len(w.audio) else w.audio
        if speech_ratio < 0.35 or len(audio) < 16000:
            return missing(Modality.VOICE_ID, w.t_start, w.t_end, "not enough speech")
        e = SpeakerEncoder.get().embed(audio)
        ref = self.enr.voice
        if ref is None:
            self.enr.auto_voice.append(e)
            if len(self.enr.auto_voice) >= self.AUTO_ENROLL_WINDOWS:
                r = np.mean(self.enr.auto_voice, 0)
                self.enr.voice = r / np.linalg.norm(r)
                self.enr.voice_source = "auto (first seconds of session)"
        sim_enr = float(ref @ e) if ref is not None else None
        sim_prev = self.voice_track.update(e)
        r_enr = _sigmoid((self.voice_thr - sim_enr) / 0.06) if sim_enr is not None else 0.0
        r_jump = _sigmoid((0.15 - sim_prev) / 0.06) if sim_prev is not None else 0.0
        risk = max(r_enr, r_jump)
        conf = float(np.clip(speech_ratio / 0.7, 0, 1)) * (0.9 if sim_enr is not None else 0.4)
        return ModuleOutput(
            module=Modality.VOICE_ID,
            score=risk,
            confidence=conf,
            t_start=w.t_start,
            t_end=w.t_end,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            details={
                "sim_enrolled": None if sim_enr is None else round(sim_enr, 3),
                "sim_running": None if sim_prev is None else round(sim_prev, 3),
                "enrollment": self.enr.voice_source or "pending",
            },
        )
