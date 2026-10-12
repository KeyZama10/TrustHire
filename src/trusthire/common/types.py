"""Shared data contracts. Every analysis module emits `ModuleOutput` per window."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Modality(StrEnum):
    VIDEO = "video_deepfake"
    AUDIO = "voice_spoof"
    AVSYNC = "av_sync"
    FACE_ID = "face_identity"
    VOICE_ID = "voice_identity"
    BEHAVIOR = "behavior"
    LIVENESS = "liveness"
    OBJECTS = "objects"
    SCRIPTED = "scripted_answer"
    RESUME = "resume_consistency"


ALL_MODALITIES: tuple[Modality, ...] = tuple(Modality)

# Grouping used for ablations ("video-only", "audio-only", ...).
MODALITY_GROUPS: dict[str, tuple[Modality, ...]] = {
    "video": (Modality.VIDEO, Modality.FACE_ID),
    "audio": (Modality.AUDIO, Modality.VOICE_ID),
    "behavior": (Modality.BEHAVIOR, Modality.LIVENESS, Modality.OBJECTS),
    "transcript": (Modality.SCRIPTED, Modality.RESUME),
    "avsync": (Modality.AVSYNC,),
}


class ModuleOutput(BaseModel):
    """Per-window output of one module.

    score: risk in [0, 1] — higher means *more suspicious* (fake / spoofed / mismatch).
    confidence: how much the module trusts this particular score in [0, 1]
                (e.g. low when no face was found, speech was absent, too few frames).
    """

    module: Modality
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    t_start: float
    t_end: float
    latency_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def timestamp(self) -> float:
        return 0.5 * (self.t_start + self.t_end)


def missing(module: Modality, t_start: float, t_end: float, reason: str) -> ModuleOutput:
    """Explicit 'no evidence' output: confidence 0, neutral score. Fusion treats it as missing."""
    return ModuleOutput(
        module=module,
        score=0.5,
        confidence=0.0,
        t_start=t_start,
        t_end=t_end,
        details={"missing": True, "reason": reason},
    )


@dataclass
class Window:
    """A 2-3 s analysis window cut by the chunker."""

    index: int
    t_start: float
    t_end: float
    frames: list[np.ndarray]  # BGR uint8 HxWx3
    frame_times: list[float]
    audio: np.ndarray  # float32 mono 16 kHz in [-1, 1]
    sample_rate: int = 16000
    cache: dict[str, Any] = field(default_factory=dict)  # shared per-window results (face dets…)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class FusedPoint(BaseModel):
    t: float
    t_start: float
    t_end: float
    risk: float
    risk_ci: tuple[float, float] | None = None
    modules: dict[str, ModuleOutput]
    contributions: dict[str, float] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)
    latency_ms: float | None = None  # wall-clock time to analyse + fuse this window


class Alert(BaseModel):
    t: float
    module: str
    severity: str  # info | warning | critical
    message: str
