"""Learn the scripted-answer detector from labelled answers of the custom test set.

The live module starts from a documented hand-set prior (features.scripted_prior_score). Once recordings with
`scripted_reading` segments exist, this fits a regularised logistic regression on the answer features
(leave-participants-out CV), reports AUC/EER against the prior, and saves models/transcript/scripted_lr.pkl,
which TranscriptMonitor uses automatically when present.

    python -m trusthire.transcript.train
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trusthire.common.registry import models_dir
from trusthire.common.settings import REPO_ROOT
from trusthire.evaluation.dataset import custom_root
from trusthire.evaluation.metrics import report

FEATURES = [
    "filler_per_100w",
    "repair_per_100w",
    "gap_cv",
    "long_pause_per_min",
    "rate_cv",
    "pitch_st_std",
    "energy_cv",
    "speech_rate_wpm",
    "mean_word_prob",
    "ai_text_prob",
    "response_latency_s",
]


def load_answers() -> pd.DataFrame:
    """Answer-level rows from the harness cache (scripted_answer details + window labels)."""
    rows = []
    for p in sorted((custom_root() / "cache").glob("*_answers.parquet")):
        rows.append(pd.read_parquet(p))
    if not rows:
        raise SystemExit(
            "no harness cache found — run `python -m trusthire.evaluation.harness extract` first"
        )
    return pd.concat(rows, ignore_index=True)


def fit(df: pd.DataFrame) -> dict:
    X = df[FEATURES].astype(float).fillna(df[FEATURES].astype(float).median())
    y = (df["attack"] == "scripted_reading").astype(int).to_numpy()
    groups = df["participant_id"].to_numpy()
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000))
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    res = {
        "learned": report(y, oof).as_dict(),
        "prior": report(y, df["scripted_answer_score"].to_numpy()).as_dict(),
    }
    final = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)
    ).fit(X, y)
    (models_dir() / "transcript").mkdir(parents=True, exist_ok=True)
    with open(models_dir() / "transcript" / "scripted_lr.pkl", "wb") as f:
        pickle.dump({"features": FEATURES, "model": final, "medians": X.median().to_dict()}, f)
    out = REPO_ROOT / "results" / "transcript"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scripted_detector.json").write_text(json.dumps(res, indent=2))
    print(f"prior AUC {res['prior']['auc']:.3f} -> learned AUC {res['learned']['auc']:.3f}")
    return res


if __name__ == "__main__":
    fit(load_answers())
