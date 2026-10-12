"""SYNTHETIC module-score sessions — for unit tests and pipeline smoke runs ONLY.

Every table produced here carries `synthetic=True`; the evaluation harness refuses to put synthetic rows
into any results table or report. Real numbers come from the custom recorded test set (docs/DATA_COLLECTION.md)
and the public-benchmark runs in results/.

Generative story: each session has an attack type; each attack shifts the score distribution of the modules
that should detect it (e.g. face_swap -> video_deepfake, av_sync), with per-module noise, missingness and
confidence, and windows outside the attack span stay genuine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trusthire.fusion.features import MODS, RAW_FEATURES

ATTACKS: dict[str, dict[str, float]] = {
    "genuine": {},
    "face_swap": {"video_deepfake": 2.2, "av_sync": 1.2, "face_identity": 1.5, "liveness": 0.6},
    "voice_clone": {"voice_spoof": 2.4, "voice_identity": 1.4, "av_sync": 0.8},
    "proxy": {"face_identity": 3.0, "voice_identity": 2.6},
    "scripted_reading": {"behavior": 1.8, "scripted_answer": 1.6},
    "second_person": {"objects": 3.0, "behavior": 0.8},
}
MISSING_P = {
    "av_sync": 0.25,
    "voice_spoof": 0.15,
    "voice_identity": 0.2,
    "liveness": 0.3,
    "scripted_answer": 0.5,
    "resume_consistency": 0.7,
}


def make_sessions(n_sessions: int = 60, windows: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    attacks = list(ATTACKS)
    for si in range(n_sessions):
        attack = attacks[si % len(attacks)]
        start = int(rng.integers(0, windows // 2)) if attack != "genuine" else windows
        for wi in range(windows):
            active = attack != "genuine" and wi >= start
            r: dict[str, object] = {
                "session_id": f"syn{si:03d}",
                "window": wi,
                "t_start": wi * 1.5,
                "t_end": wi * 1.5 + 3,
                "attack": attack if active else "genuine",
                "label": int(active),
                "synthetic": True,
            }
            for m in MODS:
                shift = ATTACKS[attack].get(m, 0.0) if active else 0.0
                z = rng.normal(-1.8 + shift * 1.6, 1.0)
                if rng.random() < MISSING_P.get(m, 0.05):
                    r[f"{m}_score"], r[f"{m}_conf"] = np.nan, 0.0
                else:
                    r[f"{m}_score"] = float(1 / (1 + np.exp(-z)))
                    r[f"{m}_conf"] = float(np.clip(rng.beta(5, 2), 0.05, 1))
            for name in RAW_FEATURES:
                r[name] = np.nan
            rows.append(r)
    return pd.DataFrame(rows)
