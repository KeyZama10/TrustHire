"""Chunker, metrics, augmentations, registry — no model weights needed."""

from __future__ import annotations

import numpy as np
import pytest

from trusthire.common import augment
from trusthire.evaluation.metrics import ece, eer, report, tpr_at_fpr
from trusthire.pipeline.chunker import SlidingChunker


def _feed(ch: SlidingChunker, seconds: float, fps: int = 25, sr: int = 16000) -> list:
    out = []
    for i in range(int(seconds * fps)):
        ch.push_frame(i / fps, np.full((4, 4, 3), i % 255, np.uint8))
        if i % 5 == 4:  # audio arrives in 200 ms blocks
            ch.push_audio(np.full(int(0.2 * sr), i, np.float32))
        out += ch.pop_windows()
    return out


def test_chunker_windows_overlap_and_lengths():
    ws = _feed(SlidingChunker(3.0, 1.5), 9.0)
    assert [w.index for w in ws] == list(range(len(ws)))
    assert len(ws) == 5  # starts at 0,1.5,3,4.5,6
    for w in ws:
        assert w.t_end - w.t_start == pytest.approx(3.0)
        assert len(w.audio) == 48000
        assert len(w.frames) == 75
        assert w.frame_times[0] >= w.t_start and w.frame_times[-1] < w.t_end
    assert ws[1].t_start - ws[0].t_start == pytest.approx(1.5)


def test_chunker_flush_emits_tail():
    ch = SlidingChunker(3.0, 1.5)
    ws = _feed(ch, 4.0)
    tail = ch.pop_windows(flush=True)
    assert len(ws) == 1 and len(tail) >= 1
    assert len(tail[0].audio) == 48000  # zero-padded


def test_chunker_rejects_bad_hop():
    with pytest.raises(ValueError):
        SlidingChunker(3.0, 4.0)


def test_eer_perfect_and_random(rng):
    y = np.r_[np.zeros(500), np.ones(500)]
    assert eer(y, y + 0.0)[0] == pytest.approx(0.0, abs=1e-9)
    e, _ = eer(y, rng.random(1000))
    assert 0.4 < e < 0.6


def test_eer_symmetric_gaussians(rng):
    y = np.r_[np.zeros(20000), np.ones(20000)]
    s = np.r_[rng.normal(0, 1, 20000), rng.normal(2, 1, 20000)]
    # theoretical EER for d'=2 is Phi(-1) = 0.1587
    assert eer(y, s)[0] == pytest.approx(0.1587, abs=0.01)


def test_tpr_at_fpr_and_report(rng):
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(3, 1, 1000)]
    tpr, prec, _ = tpr_at_fpr(y, s, 0.05)
    assert 0.8 < tpr <= 1.0 and prec > 0.9
    r = report(y, 1 / (1 + np.exp(-(s - 1.5))), bootstrap=50)
    assert r.auc > 0.95 and r.auc_ci[0] <= r.auc <= r.auc_ci[1]


def test_ece_perfectly_calibrated(rng):
    p = rng.random(50000)
    y = (rng.random(50000) < p).astype(int)
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p * 0.3, 0, 1)) > 0.2


def test_report_needs_two_classes():
    with pytest.raises(ValueError):
        report(np.zeros(10), np.zeros(10))


@pytest.mark.parametrize("cond", list(augment.AUDIO_CONDITIONS))
def test_audio_conditions_preserve_length(cond, rng):
    x = (np.sin(np.linspace(0, 400 * np.pi, 32000)) * 0.3).astype(np.float32)
    y = augment.AUDIO_CONDITIONS[cond](x, rng)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()


def test_noise_snr(rng):
    x = (np.sin(np.linspace(0, 400 * np.pi, 32000))).astype(np.float32)
    y = augment.add_noise(x, 10, rng)
    snr = 10 * np.log10(np.mean(x**2) / np.mean((y - x) ** 2))
    assert snr == pytest.approx(10, abs=0.5)


@pytest.mark.parametrize("cond", list(augment.VIDEO_CONDITIONS))
def test_video_conditions(cond):
    img = (np.random.default_rng(1).random((120, 160, 3)) * 255).astype(np.uint8)
    out = augment.VIDEO_CONDITIONS[cond](img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_random_image_aug_shape(rng):
    img = np.zeros((64, 64, 3), np.uint8)
    for _ in range(10):
        assert augment.random_image_aug(img, rng).shape == img.shape


def test_registry_lists_assets():
    from trusthire.common.registry import ASSETS, BY_KEY

    assert {"video.effnb4", "audio.aasist", "avsync.syncnet", "identity.buffalo_l"} <= set(BY_KEY)
    assert all(a.license for a in ASSETS)
