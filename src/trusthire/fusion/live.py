"""Online fusion used by the live/offline pipeline: per-window fused risk, contributions, alerts, trust score."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trusthire.common.registry import models_dir
from trusthire.common.types import Alert, FusedPoint, Modality, ModuleOutput
from trusthire.fusion.features import row_from_outputs
from trusthire.fusion.models import CalibratedFuser, WeightedLateFusion

log = logging.getLogger(__name__)

ALERT_RULES: list[tuple[Modality, float, float, str, str]] = [
    # module, min score, min confidence, severity, message
    (Modality.FACE_ID, 0.80, 0.5, "critical", "Face does not match the enrolled candidate"),
    (Modality.VOICE_ID, 0.80, 0.4, "critical", "Voice does not match the enrolled voiceprint"),
    (Modality.VIDEO, 0.75, 0.4, "critical", "Possible face manipulation (deepfake / face swap)"),
    (Modality.AUDIO, 0.80, 0.4, "critical", "Possible synthetic or cloned voice"),
    (Modality.AVSYNC, 0.75, 0.4, "warning", "Lip movements out of sync with speech"),
    (Modality.OBJECTS, 0.65, 0.5, "warning", "Suspicious object or person in frame"),
    (Modality.BEHAVIOR, 0.70, 0.5, "warning", "Gaze pattern consistent with reading from another screen"),
    (Modality.LIVENESS, 0.85, 0.4, "info", "Weak physiological (pulse) signal"),
    (Modality.SCRIPTED, 0.75, 0.4, "warning", "Answer delivery resembles reading or AI-scripted content"),
    (Modality.RESUME, 0.60, 0.4, "warning", "Answer contradicts the candidate's resume"),
]


class LiveFuser:
    def __init__(self, method: str = "weighted", alert_threshold: float = 0.65):
        self.method = method
        self.alert_threshold = alert_threshold
        path = models_dir() / "fusion" / f"{method}.pkl"
        self.trained = False
        if path.exists():
            try:
                self.model = CalibratedFuser.load(path)
                self.trained = True
            except Exception:
                log.exception("could not load %s; falling back to prior weights", path)
                self.model = CalibratedFuser(WeightedLateFusion())
        else:
            self.model = CalibratedFuser(WeightedLateFusion())
        self.history: list[dict[str, float]] = []
        self.session_id = "live"

    def fuse(
        self, outputs: dict[str, ModuleOutput], t_start: float, t_end: float, index: int | None = None
    ) -> FusedPoint:
        row = row_from_outputs(outputs)
        row["session_id"] = self.session_id  # type: ignore[assignment]
        if index is not None and index < len(self.history):
            self.history[index] = row
        else:
            self.history.append(row)
        # GBM / attention use causal temporal context, so give them the recent history
        k = index + 1 if index is not None else len(self.history)
        df = pd.DataFrame(self.history[max(0, k - 6) : k])
        risk = float(self.model.predict_proba(df)[-1])
        try:
            contrib = self.model.contributions(df).iloc[-1].to_dict()
        except Exception:
            contrib = {}
        alerts = [
            msg
            for m, smin, cmin, _, msg in ALERT_RULES
            if (o := outputs.get(m.value)) is not None
            and o.score >= smin
            and o.confidence >= cmin
            and not o.details.get("missing")
        ]
        if risk >= self.alert_threshold and not alerts:
            alerts.append("Elevated overall risk")
        return FusedPoint(
            t=0.5 * (t_start + t_end),
            t_start=t_start,
            t_end=t_end,
            risk=risk,
            modules=outputs,
            contributions={k: round(float(v), 4) for k, v in contrib.items()},
            alerts=alerts,
        )


def alerts_from_point(p: FusedPoint) -> list[Alert]:
    out = []
    for m, smin, cmin, sev, msg in ALERT_RULES:
        o = p.modules.get(m.value)
        if o is not None and o.score >= smin and o.confidence >= cmin and not o.details.get("missing"):
            out.append(Alert(t=p.t, module=m.value, severity=sev, message=msg))
    return out


def session_trust(points: list[FusedPoint]) -> dict[str, float]:
    """Trust score 0–100. Robust session risk = 0.6·mean(top-10% window risks) + 0.4·mean(risk),
    so a short burst of strong evidence matters but a single noisy window does not decide alone."""
    if not points:
        return {"trust_score": float("nan"), "session_risk": float("nan"), "peak_risk": float("nan")}
    r = np.array([p.risk for p in points])
    k = max(1, int(np.ceil(0.1 * len(r))))
    top = np.sort(r)[-k:].mean()
    risk = 0.6 * top + 0.4 * r.mean()
    return {
        "trust_score": round(100 * (1 - risk), 1),
        "session_risk": round(float(risk), 4),
        "peak_risk": round(float(r.max()), 4),
        "mean_risk": round(float(r.mean()), 4),
        "flagged_fraction": round(float((r >= 0.65).mean()), 4),
    }
