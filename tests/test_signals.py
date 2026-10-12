"""Behaviour (rPPG, reading sweeps, pose), transcript features and resume chunking — signal-level tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trusthire.behavior.infer import euler_from_matrix, reading_sweeps
from trusthire.behavior.rppg import chrom, pos, pulse_snr
from trusthire.transcript.asr import Word
from trusthire.transcript.features import compute, filler_count, scripted_prior_score
from trusthire.transcript.nlp import chunk_resume


def _skin(
    hr_bpm: float, fs: float = 25, secs: float = 10, amp: float = 0.6, noise: float = 0.3, seed: int = 0
):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    pulse = np.sin(2 * np.pi * hr_bpm / 60 * t)
    base = np.array([150.0, 110.0, 90.0])
    # blood volume pulse modulates green most (as in real skin), plus sensor noise
    rgb = base + np.outer(pulse, [0.3 * amp, amp, 0.2 * amp]) + rng.normal(0, noise, (len(t), 3))
    return rgb, fs


@pytest.mark.parametrize("hr", [60, 75, 110])
@pytest.mark.parametrize("algo", [pos, chrom])
def test_rppg_recovers_heart_rate(hr, algo):
    rgb, fs = _skin(hr)
    est, snr = pulse_snr(algo(rgb, fs), fs)
    assert est == pytest.approx(hr, abs=4)
    assert snr > 0


def test_rppg_flat_signal_low_snr():
    rng = np.random.default_rng(1)
    rgb = np.array([150.0, 110.0, 90.0]) + rng.normal(0, 0.3, (250, 3))
    _, snr_noise = pulse_snr(pos(rgb, 25), 25)
    _, snr_live = pulse_snr(pos(_skin(72)[0], 25), 25)
    assert snr_live > snr_noise + 3


def test_reading_sweeps_detects_line_reading():
    ts = np.arange(0, 10, 0.12)
    saw = np.concatenate([np.linspace(-0.4, 0.4, 12) for _ in range(len(ts) // 12 + 1)])[: len(ts)]
    still = np.random.default_rng(0).normal(0, 0.01, len(ts))
    assert reading_sweeps(saw, ts) > 0.3
    assert reading_sweeps(still, ts) < 0.1


def test_euler_identity_and_yaw():
    assert euler_from_matrix(np.eye(4)) == pytest.approx((0, 0, 0), abs=1e-6)
    a = math.radians(30)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    assert euler_from_matrix(m)[0] == pytest.approx(30, abs=1e-4)


def _words(text: str, gap: float = 0.15, jitter: float = 0.0, seed: int = 0) -> list[Word]:
    rng = np.random.default_rng(seed)
    out, t = [], 0.0
    for w in text.split():
        d = 0.25 + 0.02 * len(w)
        out.append(Word(t, t + d, w, 0.9))
        t += d + max(0.02, gap + rng.normal(0, jitter))
    return out


def test_filler_count():
    fill, reps = filler_count(_words("um so I I think uh it was like you know fine"))
    assert fill >= 3 and reps == 1


def test_features_and_prior_order():
    spontaneous = _words(
        "um so I think uh the project was like kind of hard I I mean we um rebuilt the pipeline "
        "and uh it worked eventually you know after some back and forth with the team",
        gap=0.35,
        jitter=0.4,
    )
    read = _words(
        "The project required a complete redesign of the data pipeline. We migrated to a streaming "
        "architecture which reduced latency by forty percent and improved reliability across services.",
        gap=0.12,
        jitter=0.02,
    )
    fs, fr = compute(spontaneous, None, 0.0, None), compute(read, None, 0.0, -2.0)
    assert fs.filler_per_100w > fr.filler_per_100w
    assert fs.gap_cv > fr.gap_cv
    assert fr.response_latency_s == pytest.approx(2.0)
    assert scripted_prior_score(fr, 0.9) > scripted_prior_score(fs, 0.1)


def test_chunk_resume():
    text = (
        "JANE DOE\nSoftware Engineer\n• Built a real-time fraud detection service in Go handling 20k rps\n"
        "• Led migration from MySQL to PostgreSQL across 40 services.\nEDUCATION\nB.Tech Computer Science, 2021"
    )
    chunks = chunk_resume(text)
    assert any("fraud detection" in c for c in chunks)
    assert any("PostgreSQL" in c for c in chunks)
    assert all(len(c) > 3 for c in chunks)
