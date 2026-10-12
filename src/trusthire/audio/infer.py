"""Per-window voice anti-spoofing inference (ONNX Runtime), gated by Silero VAD."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from trusthire.common.ort import make_session
from trusthire.common.registry import ensure, models_dir
from trusthire.common.types import Modality, ModuleOutput, Window, missing

NB_SAMP = 64600
SR = 16000


def speech_ratio(audio: np.ndarray, sr: int = SR) -> float:
    """Fraction of samples inside Silero-VAD speech segments (faster-whisper bundles the ONNX model)."""
    if audio.size == 0 or float(np.abs(audio).max()) < 1e-4:
        return 0.0
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    segs = get_speech_timestamps(
        audio, VadOptions(min_silence_duration_ms=200, speech_pad_ms=30), sampling_rate=sr
    )
    return float(sum(s["end"] - s["start"] for s in segs) / len(audio))


def speech_snr_db(audio: np.ndarray, sr: int = SR) -> float:
    """Blind SNR estimate by minimum statistics: 20 ms frame powers, speech = 90th pct, noise = 10th pct."""
    hop = sr // 50
    n = len(audio) // hop
    if n < 10:
        return float("nan")
    pw = (audio[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12
    return float(10 * np.log10(np.percentile(pw, 90) / np.percentile(pw, 10)))


def tile(x: np.ndarray, n: int = NB_SAMP) -> np.ndarray:
    if len(x) >= n:
        return x[-n:]
    return np.tile(x, int(np.ceil(n / max(1, len(x)))))[:n]


def preemphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.concatenate([x[:1], x[1:] - coef * x[:-1]]).astype(np.float32)


class _Backend:
    name = "base"

    def p_spoof(self, wavs: np.ndarray) -> np.ndarray:  # (B, NB_SAMP) -> (B,)
        raise NotImplementedError


def _softmax_spoof(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e[:, 0] / e.sum(1))  # index 0 = spoof, 1 = bona fide


class AASISTBackend(_Backend):
    def __init__(self, variant: str = "aasist"):
        self.name = variant
        p = models_dir() / "audio" / f"{variant}.onnx"
        if not p.exists():
            from trusthire.audio.model import build_aasist
            from trusthire.common.export import export_onnx

            w = ensure("audio.aasist" if variant == "aasist" else "audio.aasist_l")
            export_onnx(
                build_aasist(variant, w), (torch.randn(2, NB_SAMP) * 0.1,), p, ["wav"], ["logits"], atol=1e-2
            )
        self.sess = make_session(p)
        self._lock = threading.Lock()

    def p_spoof(self, wavs: np.ndarray) -> np.ndarray:
        with self._lock:
            logits = self.sess.run(None, {"wav": wavs.astype(np.float32)})[0]
        return _softmax_spoof(logits)


class Spectra0Backend(_Backend):
    """lab260/spectra_0 (XLS-R-300M + ECAPA). Input: pre-emphasised 64600-sample waveform."""

    name = "spectra0"

    def __init__(self, path: Path | None = None):
        p = path or models_dir() / "audio" / "spectra0" / "spectra-0.onnx"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run `trusthire download-models --only audio.spectra0`")
        self.sess = make_session(p)
        self.inp = self.sess.get_inputs()[0]
        self._lock = threading.Lock()

    def p_spoof(self, wavs: np.ndarray) -> np.ndarray:
        x = np.stack([preemphasis(w) for w in wavs]).astype(np.float32)
        out = []
        with self._lock:
            fixed_batch = isinstance(self.inp.shape[0], int)
            batches = [x[i : i + 1] for i in range(len(x))] if fixed_batch else [x]
            for b in batches:
                out.append(self.sess.run(None, {self.inp.name: b})[0])
        return _softmax_spoof(np.concatenate(out))


def spectra0_int8_path() -> Path:
    """Dynamic int8 quantisation of Spectra-0 (weights only) for CPU deployment; created on first use."""
    src = models_dir() / "audio" / "spectra0" / "spectra-0.onnx"
    dst = src.with_name("spectra-0.int8.onnx")
    if not dst.exists():
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"]
        )
    return dst


def dereverb(x: np.ndarray) -> np.ndarray:
    """WPE dereverberation (Nakatani et al. 2010; nara_wpe), single channel, ~0.3 s per 4 s on CPU."""
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    y = stft(x[None].astype(np.float64), size=512, shift=128).transpose(2, 0, 1)
    z = wpe(y, taps=10, delay=3, iterations=3, statistics_mode="full").transpose(1, 2, 0)
    out = istft(z, size=512, shift=128)[0][: len(x)]
    out = np.pad(out, (0, len(x) - len(out)))
    return (out / (np.abs(out).max() + 1e-9) * (np.abs(x).max() + 1e-9)).astype(np.float32)


class DereverbBackend(_Backend):
    """Any backend with a WPE front-end (evaluated in results/audio: reverb robustness)."""

    def __init__(self, inner: _Backend):
        self.inner = inner
        self.name = f"{inner.name}+wpe"

    def p_spoof(self, wavs: np.ndarray) -> np.ndarray:
        return self.inner.p_spoof(np.stack([dereverb(w) for w in wavs]))


def build_backend(name: str) -> _Backend:
    if name.endswith("+wpe"):
        return DereverbBackend(build_backend(name.removesuffix("+wpe")))
    if name in ("aasist", "aasist_l"):
        return AASISTBackend(name)
    if name == "spectra0":
        return Spectra0Backend()
    if name == "spectra0_int8":
        be = Spectra0Backend(spectra0_int8_path())
        be.name = "spectra0_int8"
        return be
    raise ValueError(name)


class VoiceSpoofDetector:
    """Scores the most recent ~4 s of speech ending at the window end.

    The analyzer passes `ctx_audio` (window audio prefixed by the preceding second) so each decision
    uses the 4.04 s context the models were trained on.
    """

    def __init__(self, backend: str = "aasist"):
        self.backend = build_backend(backend)

    def process(self, w: Window, ctx_audio: np.ndarray | None = None) -> ModuleOutput:
        t0 = time.perf_counter()
        audio = ctx_audio if ctx_audio is not None and len(ctx_audio) >= len(w.audio) else w.audio
        sr_ratio = speech_ratio(w.audio)
        if sr_ratio < 0.15:
            return missing(Modality.AUDIO, w.t_start, w.t_end, "no speech")
        p = float(self.backend.p_spoof(tile(audio)[None])[0])
        snr = speech_snr_db(audio)
        # Measured: genuine speech is falsely flagged 5 % clean vs 17 % at 10 dB SNR (results/audio), so the
        # module reports lower confidence in noisy windows and fusion down-weights it.
        snr_conf = float(np.clip((snr - 8.0) / 17.0, 0.25, 1.0)) if np.isfinite(snr) else 0.5
        conf = float(np.clip(sr_ratio / 0.6, 0.0, 1.0)) * snr_conf
        return ModuleOutput(
            module=Modality.AUDIO,
            score=p,
            confidence=conf,
            t_start=w.t_start,
            t_end=w.t_end,
            latency_ms=(time.perf_counter() - t0) * 1e3,
            details={
                "backend": self.backend.name,
                "speech_ratio": round(sr_ratio, 3),
                "snr_db": round(snr, 1) if np.isfinite(snr) else None,
            },
        )
