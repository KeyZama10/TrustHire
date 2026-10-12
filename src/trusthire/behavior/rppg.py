"""Remote photoplethysmography (rPPG) liveness: POS (Wang et al. 2017) and CHROM (de Haan & Jeanne 2013).

A live face shows a periodic skin-colour pulse in 0.7–3 Hz; a replayed/synthesised face rendered by a
face-swap pipeline usually has a much weaker, less peaked spectrum. We report HR and spectral SNR.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch


def pos(rgb: np.ndarray, fs: float) -> np.ndarray:
    """rgb: (T,3) mean skin colour per frame -> pulse signal (T,)."""
    t = len(rgb)
    win = int(1.6 * fs)
    h = np.zeros(t)
    proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
    for n in range(win, t + 1):
        c = rgb[n - win : n].T
        mu = c.mean(1, keepdims=True)
        if np.any(mu <= 0):
            continue
        cn = c / mu
        s = proj @ cn
        sd1, sd2 = s[0].std(), s[1].std()
        p = s[0] + (sd1 / (sd2 + 1e-9)) * s[1]
        h[n - win : n] += p - p.mean()
    return h


def chrom(rgb: np.ndarray, fs: float) -> np.ndarray:
    c = rgb / (rgb.mean(0, keepdims=True) + 1e-9)
    x = 3 * c[:, 0] - 2 * c[:, 1]
    y = 1.5 * c[:, 0] + c[:, 1] - 1.5 * c[:, 2]
    b, a = butter(3, [0.7 / (fs / 2), 3.0 / (fs / 2)], btype="band")
    xf, yf = filtfilt(b, a, x), filtfilt(b, a, y)
    return np.asarray(xf - (xf.std() / (yf.std() + 1e-9)) * yf)


def pulse_snr(sig: np.ndarray, fs: float) -> tuple[float, float]:
    """(heart rate bpm, SNR dB) — power at the peak ±0.1 Hz and its 1st harmonic vs the rest of 0.7–4 Hz."""
    sig = detrend(sig)
    if len(sig) < fs * 4 or np.allclose(sig, 0):
        return float("nan"), float("-inf")
    b, a = butter(3, [0.7 / (fs / 2), min(4.0 / (fs / 2), 0.99)], btype="band")
    sig = filtfilt(b, a, sig)
    f, p = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 8)), nfft=2048)
    band = (f >= 0.7) & (f <= 3.0)
    if not band.any():
        return float("nan"), float("-inf")
    f0 = float(f[band][np.argmax(p[band])])
    sigmask = (np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.2)
    full = (f >= 0.7) & (f <= 4.0)
    ps = p[sigmask & full].sum()
    pn = p[full & ~sigmask].sum() + 1e-12
    return f0 * 60.0, float(10 * np.log10(ps / pn + 1e-12))
