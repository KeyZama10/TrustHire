"""Detection metrics. Convention: y=1 is the positive (fake / spoof / fraudulent) class; higher score = more suspicious."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def eer(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where FPR == FNR (linear interpolation on the ROC)."""
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    if i + 1 < len(fpr) and (fnr[i] - fpr[i]) * (fnr[i + 1] - fpr[i + 1]) < 0:
        a = (fnr[i] - fpr[i]) / ((fnr[i] - fpr[i]) - (fnr[i + 1] - fpr[i + 1]))
        e = fpr[i] + a * (fpr[i + 1] - fpr[i])
        t = thr[i] + a * (thr[i + 1] - thr[i]) if np.isfinite(thr[i]) else thr[i + 1]
        return float(e), float(t)
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float) -> tuple[float, float, float]:
    """(recall, precision, threshold) at the largest threshold whose FPR <= target."""
    fpr, tpr, thr = roc_curve(y, s)
    ok = np.where(fpr <= target_fpr)[0]
    i = int(ok[-1]) if len(ok) else 0
    t = thr[i]
    pred = s >= t
    prec = float(precision_score(y, pred, zero_division=0))
    return float(tpr[i]), prec, float(t)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of probabilities p for the positive class (equal-width bins)."""
    p = np.clip(p, 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(total)


@dataclass
class DetectionReport:
    n: int
    n_pos: int
    auc: float
    eer: float
    eer_threshold: float
    f1_at_eer: float
    f1_at_05: float
    tpr_at_fpr1: float
    prec_at_fpr1: float
    tpr_at_fpr5: float
    prec_at_fpr5: float
    ece: float | None
    auc_ci: tuple[float, float] | None = None
    eer_ci: tuple[float, float] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def report(
    y: np.ndarray, s: np.ndarray, probabilistic: bool = True, bootstrap: int = 0, seed: int = 0
) -> DetectionReport:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if len(np.unique(y)) < 2:
        raise ValueError("need both classes to compute detection metrics")
    e, t = eer(y, s)
    r1, p1, _ = tpr_at_fpr(y, s, 0.01)
    r5, p5, _ = tpr_at_fpr(y, s, 0.05)
    rep = DetectionReport(
        n=len(y),
        n_pos=int(y.sum()),
        auc=float(roc_auc_score(y, s)),
        eer=e,
        eer_threshold=t,
        f1_at_eer=float(f1_score(y, s >= t, zero_division=0)),
        f1_at_05=float(f1_score(y, s >= 0.5, zero_division=0)) if probabilistic else float("nan"),
        tpr_at_fpr1=r1,
        prec_at_fpr1=p1,
        tpr_at_fpr5=r5,
        prec_at_fpr5=p5,
        ece=ece(y, s) if probabilistic else None,
    )
    if bootstrap:
        rng = np.random.default_rng(seed)
        aucs, eers = [], []
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        for _ in range(bootstrap):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])  # stratified
            aucs.append(roc_auc_score(y[idx], s[idx]))
            eers.append(eer(y[idx], s[idx])[0])
        rep.auc_ci = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
        rep.eer_ci = (float(np.percentile(eers, 2.5)), float(np.percentile(eers, 97.5)))
    return rep


def recall_precision_at(y: np.ndarray, s: np.ndarray, thr: float) -> tuple[float, float]:
    pred = np.asarray(s) >= thr
    return float(recall_score(y, pred, zero_division=0)), float(precision_score(y, pred, zero_division=0))
