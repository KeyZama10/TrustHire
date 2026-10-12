"""Degradations used both for training augmentation and for robustness evaluation.

Audio: additive noise at SNR, lossy codecs (Opus / MP3 / AAC via PyAV), telephone band-limiting, reverb.
Video: JPEG/H.264-like compression, blur, low resolution (webcam quality), brightness/gamma (lighting).
"""

from __future__ import annotations

import io
from typing import Any

import av
import cv2
import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt

SR = 16000


# ----------------------------------------------------------------------------- audio
def add_noise(
    x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None, kind: str = "white"
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = rng.standard_normal(len(x)).astype(np.float32)
    if kind == "pink":
        f = np.fft.rfft(n)
        f /= np.sqrt(np.arange(1, len(f) + 1))
        n = np.fft.irfft(f, n=len(x)).astype(np.float32)
    px = float(np.mean(x**2)) + 1e-12
    pn = float(np.mean(n**2)) + 1e-12
    return (x + n * np.sqrt(px / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def codec(x: np.ndarray, fmt: str = "opus", bitrate: int = 16000, sr: int = SR) -> np.ndarray:
    """Round-trip through a lossy codec in memory (Opus @48 kHz like WebRTC, MP3, AAC)."""
    codec_name, container, rate = {
        "opus": ("libopus", "ogg", 48000),
        "mp3": ("libmp3lame", "mp3", 16000),
        "aac": ("aac", "adts", 16000),
    }[fmt]
    y = resample_poly(x, rate // 1000, sr // 1000).astype(np.float32) if rate != sr else x
    buf = io.BytesIO()
    with av.open(buf, "w", format=container) as out:
        st: Any = out.add_stream(codec_name, rate=rate)  # AudioStream (PyAV stubs return a union)
        st.bit_rate = bitrate
        st.layout = "mono"
        frame = av.AudioFrame.from_ndarray(y[None, :].astype(np.float32), format="flt", layout="mono")
        frame.sample_rate = rate
        frame.pts = 0
        for p in st.encode(frame):
            out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    buf.seek(0)
    res = av.AudioResampler(format="flt", layout="mono", rate=sr)
    chunks = []
    with av.open(buf, "r") as c:
        for fr in c.decode(audio=0):
            for f in res.resample(fr):
                chunks.append(f.to_ndarray().reshape(-1))
    z = np.concatenate(chunks) if chunks else np.zeros_like(x)
    z = z[: len(x)]
    return np.pad(z, (0, len(x) - len(z))).astype(np.float32)


def telephone(x: np.ndarray, sr: int = SR) -> np.ndarray:
    sos = butter(6, [300, 3400], btype="band", fs=sr, output="sos")
    y = sosfilt(sos, x)
    return resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)].astype(np.float32)


def reverb(
    x: np.ndarray, rt60: float = 0.4, sr: int = SR, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    rir /= np.abs(rir).sum() ** 0.5
    y = fftconvolve(x, rir)[: len(x)]
    return (y / (np.abs(y).max() + 1e-9) * (np.abs(x).max() + 1e-9)).astype(np.float32)


AUDIO_CONDITIONS = {
    "clean": lambda x, r: x,
    "noise_snr10": lambda x, r: add_noise(x, 10, r),
    "reverb_0.25": lambda x, r: reverb(x, 0.25, rng=r),
    "reverb_0.7": lambda x, r: reverb(x, 0.7, rng=r),
    "noise_snr20_pink": lambda x, r: add_noise(x, 20, r, "pink"),
    "opus_16k": lambda x, r: codec(x, "opus", 16000),
    "mp3_32k": lambda x, r: codec(x, "mp3", 32000),
    "telephone": lambda x, r: telephone(x),
    "reverb_0.4": lambda x, r: reverb(x, 0.4, rng=r),
}


def random_audio_aug(x: np.ndarray, rng: np.random.Generator, p: float = 0.7) -> np.ndarray:
    if rng.random() > p:
        return x
    ops = ["noise", "codec", "telephone", "reverb", "gain"]
    for op in rng.choice(ops, size=rng.integers(1, 3), replace=False):
        if op == "noise":
            x = add_noise(x, float(rng.uniform(5, 30)), rng, "pink" if rng.random() < 0.5 else "white")
        elif op == "codec":
            x = codec(
                x, str(rng.choice(["opus", "mp3", "aac"])), int(rng.choice([12000, 16000, 24000, 32000]))
            )
        elif op == "telephone":
            x = telephone(x)
        elif op == "reverb":
            x = reverb(x, float(rng.uniform(0.2, 0.8)), rng=rng)
        else:
            x = (x * rng.uniform(0.3, 1.5)).clip(-1, 1).astype(np.float32)
    return x


# ----------------------------------------------------------------------------- video / image
def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else None
    return out if out is not None else img


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w / factor)), max(8, int(h / factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


VIDEO_CONDITIONS = {
    "clean": lambda im: im,
    "jpeg_q30": lambda im: jpeg(im, 30),
    "jpeg_q60": lambda im: jpeg(im, 60),
    "lowres_x3": lambda im: downscale(im, 3),
    "blur_s2": lambda im: gaussian_blur(im, 2),
    "dark_g2.2": lambda im: gamma(im, 2.2),
    "bright_g0.5": lambda im: gamma(im, 0.5),
}


def random_image_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Heavy augmentation for face-forgery training (compression, blur, colour, resolution, flip)."""
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
    if rng.random() < 0.6:
        img = jpeg(img, int(rng.integers(30, 96)))
    if rng.random() < 0.3:
        img = gaussian_blur(img, float(rng.uniform(0.3, 2.0)))
    if rng.random() < 0.3:
        img = downscale(img, float(rng.uniform(1.2, 3.0)))
    if rng.random() < 0.5:
        img = gamma(img, float(rng.uniform(0.6, 1.6)))
    if rng.random() < 0.4:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 180
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img
