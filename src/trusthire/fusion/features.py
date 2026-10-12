"""Turn per-window module outputs into fusion feature matrices.

Table layout (one row per window): session_id, window, t_start, t_end, label, attack,
  <module>_score, <module>_conf for every module (score NaN when the module had no evidence),
  plus a few raw module measurements that are more informative than the hand-set risk priors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from trusthire.common.types import ALL_MODALITIES, Modality, ModuleOutput

MODS: tuple[str, ...] = tuple(m.value for m in ALL_MODALITIES)

RAW_FEATURES: dict[str, tuple[str, str]] = {
    "sync_conf": (Modality.AVSYNC.value, "sync_conf"),
    "sync_abs_offset": (Modality.AVSYNC.value, "offset_frames"),
    "face_sim_enrolled": (Modality.FACE_ID.value, "sim_enrolled"),
    "face_sim_running": (Modality.FACE_ID.value, "sim_running"),
    "voice_sim_enrolled": (Modality.VOICE_ID.value, "sim_enrolled"),
    "voice_sim_running": (Modality.VOICE_ID.value, "sim_running"),
    "rppg_snr": (Modality.LIVENESS.value, "snr_db"),
    "reading_sweeps": (Modality.BEHAVIOR.value, "reading_sweeps_per_s"),
    "off_axis_frac": (Modality.BEHAVIOR.value, "off_axis_frac"),
    "filler_rate": (Modality.SCRIPTED.value, "filler_per_100w"),
    "gap_cv": (Modality.SCRIPTED.value, "gap_cv"),
    "ai_text_prob": (Modality.SCRIPTED.value, "ai_text_prob"),
}


def row_from_outputs(outputs: dict[str, ModuleOutput]) -> dict[str, float]:
    r: dict[str, float] = {}
    for m in MODS:
        o = outputs.get(m)
        has = o is not None and o.confidence > 0 and not o.details.get("missing")
        r[f"{m}_score"] = float(o.score) if has and o is not None else np.nan
        r[f"{m}_conf"] = float(o.confidence) if has and o is not None else 0.0
    for name, (m, key) in RAW_FEATURES.items():
        o = outputs.get(m)
        v = o.details.get(key) if o is not None else None
        if name == "sync_abs_offset" and v is not None:
            v = abs(v)
        r[name] = float(v) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else np.nan
    return r


def score_cols(mods: Sequence[str] = MODS) -> list[str]:
    return [f"{m}_score" for m in mods]


def conf_cols(mods: Sequence[str] = MODS) -> list[str]:
    return [f"{m}_conf" for m in mods]


def raw_cols(mods: Sequence[str] = MODS) -> list[str]:
    return [n for n, (m, _) in RAW_FEATURES.items() if m in mods]


def add_temporal_context(df: pd.DataFrame, mods: Sequence[str] = MODS, k: int = 5) -> pd.DataFrame:
    """Rolling mean/max of each module score over the previous k windows within a session (causal)."""
    out = df.copy()
    g = out.groupby("session_id", sort=False)
    for c in score_cols(mods):
        out[c + "_rmean"] = g[c].transform(lambda s: s.rolling(k, min_periods=1).mean())
        out[c + "_rmax"] = g[c].transform(lambda s: s.rolling(k, min_periods=1).max())
    return out


def mask_modalities(df: pd.DataFrame, keep: Iterable[str]) -> pd.DataFrame:
    """Ablation helper: drop every module not in `keep` (as if it never produced evidence)."""
    keep = set(keep)
    out = df.copy()
    for m in MODS:
        if m not in keep:
            out[f"{m}_score"] = np.nan
            out[f"{m}_conf"] = 0.0
    for name, (m, _) in RAW_FEATURES.items():
        if m not in keep:
            out[name] = np.nan
    return out
