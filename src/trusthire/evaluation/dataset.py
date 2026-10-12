"""Custom interview test set: schema, validation and loader (see docs/DATA_COLLECTION.md).

Layout (default root: ~/.trusthire/data/custom, override with TRUSTHIRE_DATA_DIR):

    custom/
      manifest.csv                 one row per recorded session
      recordings/<session_id>.mp4  the interview as the interviewer's client received it
      enroll/<participant>_id.jpg  ID / reference photo     (optional per row)
      enroll/<participant>_voice.wav  10 s voice sample     (optional per row)
      resumes/<participant>.pdf    resume                   (optional)
      labels/<session_id>.json     time-stamped segment labels

labels/<session_id>.json:
    {"session_id": "s012", "segments": [
        {"start": 0.0,  "end": 95.0,  "label": "genuine"},
        {"start": 95.0, "end": 180.0, "label": "face_swap", "tool": "DeepFaceLive", "notes": "..."}
    ]}
"""

from __future__ import annotations

import itertools
import json
from enum import StrEnum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from trusthire.common.registry import data_dir


class Attack(StrEnum):
    GENUINE = "genuine"
    FACE_SWAP = "face_swap"  # real-time face swap (DeepFaceLive / similar)
    LIP_SYNC = "lip_sync"  # talking-head / lip-sync puppet (e.g. LivePortrait/Wav2Lip driven)
    VOICE_CLONE = "voice_clone"  # TTS / voice conversion (RVC, OpenVoice, XTTS ...)
    FACE_AND_VOICE = "face_and_voice"
    PROXY = "proxy"  # a different (real) person after enrollment
    SCRIPTED_READING = "scripted_reading"  # reading prepared / AI-generated answers from a screen
    SECOND_PERSON = "second_person"  # someone else in the room helping / whispering
    PHONE_USE = "phone_use"


MANIFEST_COLUMNS = [
    "session_id",
    "participant_id",
    "recording",
    "attack",
    "id_photo",
    "voice_sample",
    "resume",
    "lighting",
    "accent",
    "webcam",
    "bitrate_kbps",
    "gender",
    "skin_tone",
    "glasses",
    "notes",
]

# metadata columns used for the fairness / robustness breakdown
STRATA = ["lighting", "accent", "webcam", "bitrate_kbps", "gender", "skin_tone", "glasses"]


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float
    label: Attack
    tool: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _order(self) -> Segment:
        if self.end <= self.start:
            raise ValueError(f"segment end {self.end} <= start {self.start}")
        return self


class SessionLabels(BaseModel):
    session_id: str
    segments: list[Segment]

    @model_validator(mode="after")
    def _no_overlap(self) -> SessionLabels:
        segs = sorted(self.segments, key=lambda s: s.start)
        for a, b in itertools.pairwise(segs):
            if b.start < a.end - 1e-6:
                raise ValueError(
                    f"{self.session_id}: overlapping segments {a.start}-{a.end} and {b.start}-{b.end}"
                )
        return self

    def label_at(self, t0: float, t1: float, min_overlap: float = 0.5) -> tuple[int, str]:
        """Window label: fraudulent (1) if >= min_overlap of the window lies in a non-genuine segment."""
        dur = t1 - t0
        best, best_ov = Attack.GENUINE.value, 0.0
        fraud = 0.0
        for s in self.segments:
            ov = max(0.0, min(t1, s.end) - max(t0, s.start))
            if s.label != Attack.GENUINE:
                fraud += ov
                if ov > best_ov:
                    best, best_ov = s.label.value, ov
        return int(fraud / dur >= min_overlap), best


def custom_root() -> Path:
    return data_dir() / "custom"


def load_manifest(root: Path | None = None) -> pd.DataFrame:
    root = root or custom_root()
    df = pd.read_csv(root / "manifest.csv", dtype=str).fillna("")
    missing = [c for c in ("session_id", "participant_id", "recording", "attack") if c not in df]
    if missing:
        raise ValueError(f"manifest.csv missing columns {missing}")
    for c in MANIFEST_COLUMNS:
        if c not in df:
            df[c] = ""
    bad = sorted(set(df.attack) - {a.value for a in Attack})
    if bad:
        raise ValueError(f"unknown attack values {bad}; allowed: {[a.value for a in Attack]}")
    if df.session_id.duplicated().any():
        raise ValueError("duplicate session_id in manifest")
    return df


def load_labels(root: Path, session_id: str) -> SessionLabels:
    return SessionLabels.model_validate(
        json.loads((root / "labels" / f"{session_id}.json").read_text(encoding="utf-8"))
    )


def validate(root: Path | None = None) -> list[str]:
    """Return a list of problems (empty == dataset is consistent)."""
    root = root or custom_root()
    problems: list[str] = []
    try:
        df = load_manifest(root)
    except Exception as e:
        return [f"manifest: {e}"]
    for _, r in df.iterrows():
        rec = root / r.recording
        if not rec.exists():
            problems.append(f"{r.session_id}: missing recording {rec}")
        for col in ("id_photo", "voice_sample", "resume"):
            if r[col] and not (root / r[col]).exists():
                problems.append(f"{r.session_id}: missing {col} {r[col]}")
        try:
            lab = load_labels(root, r.session_id)
            kinds = {s.label.value for s in lab.segments}
            if r.attack != "genuine" and r.attack not in kinds:
                problems.append(f"{r.session_id}: manifest attack '{r.attack}' never appears in labels")
        except FileNotFoundError:
            problems.append(f"{r.session_id}: no labels/{r.session_id}.json")
        except Exception as e:
            problems.append(f"{r.session_id}: bad labels ({e})")
    return problems


def write_template(root: Path | None = None) -> Path:
    """Create an empty dataset skeleton with one example row + label file."""
    root = root or custom_root()
    for d in ("recordings", "enroll", "resumes", "labels"):
        (root / d).mkdir(parents=True, exist_ok=True)
    m = root / "manifest.csv"
    if not m.exists():
        pd.DataFrame(
            [
                {
                    "session_id": "s001",
                    "participant_id": "p01",
                    "recording": "recordings/s001.mp4",
                    "attack": "face_swap",
                    "id_photo": "enroll/p01_id.jpg",
                    "voice_sample": "enroll/p01_voice.wav",
                    "resume": "resumes/p01.pdf",
                    "lighting": "normal",
                    "accent": "indian-english",
                    "webcam": "720p-laptop",
                    "bitrate_kbps": "1500",
                    "gender": "f",
                    "skin_tone": "IV",
                    "glasses": "no",
                    "notes": "",
                }
            ],
            columns=MANIFEST_COLUMNS,
        ).to_csv(m, index=False)
        (root / "labels" / "s001.json").write_text(
            json.dumps(
                {
                    "session_id": "s001",
                    "segments": [
                        {"start": 0, "end": 90, "label": "genuine"},
                        {"start": 90, "end": 240, "label": "face_swap", "tool": "DeepFaceLive"},
                    ],
                },
                indent=2,
            )
        )
    return root
