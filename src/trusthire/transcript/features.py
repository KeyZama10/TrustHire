"""Answer-level features separating spontaneous speech from read / AI-scripted answers.

Read or recited speech is more fluent than spontaneous speech: fewer filled pauses and repairs, more
regular pausing (lower inter-word-gap variability), steadier speaking rate, flatter pitch, and often a
suspiciously short or suspiciously long-then-fluent response latency.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass

import numpy as np

from trusthire.transcript.asr import Word

FILLERS = {"um", "umm", "uh", "uhh", "erm", "er", "ah", "hmm", "mm", "mhm", "eh"}
SOFT_FILLERS = {"like", "basically", "actually", "literally"}
FILLER_PHRASES = ("you know", "i mean", "kind of", "sort of", "let me think")
_TOK = re.compile(r"[a-z']+")


@dataclass
class AnswerFeatures:
    n_words: int
    duration_s: float
    speech_rate_wpm: float
    filler_per_100w: float
    repair_per_100w: float
    gap_cv: float  # coefficient of variation of inter-word gaps
    long_pause_per_min: float
    rate_cv: float  # variability of local speaking rate
    pitch_st_std: float | None  # F0 std in semitones
    energy_cv: float
    response_latency_s: float | None
    mean_word_prob: float

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def filler_count(words: list[Word]) -> tuple[int, int]:
    toks = [t for w in words for t in tokens(w.text)]
    hard = sum(t in FILLERS for t in toks)
    soft = sum(t in SOFT_FILLERS for t in toks)
    joined = " ".join(toks)
    phr = sum(joined.count(p) for p in FILLER_PHRASES)
    # repairs: immediate word repetitions ("I I think", "the the")
    reps = sum(1 for a, b in itertools.pairwise(toks) if a == b and a not in FILLERS)
    return hard + phr + soft // 2, reps


def pitch_semitone_std(audio: np.ndarray, sr: int = 16000) -> float | None:
    """F0 variability in semitones. Praat (parselmouth) if loadable, else librosa YIN with voicing gate."""
    if len(audio) < sr:
        return None
    f0: np.ndarray
    try:
        import parselmouth

        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        f0 = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
        f0 = f0[f0 > 0]
    except Exception:  # parselmouth missing or blocked by OS policy
        import librosa

        y = librosa.yin(audio.astype(np.float32), fmin=75, fmax=400, sr=sr, frame_length=1024, hop_length=320)
        rms = librosa.feature.rms(y=audio.astype(np.float32), frame_length=1024, hop_length=320)[0][: len(y)]
        f0 = y[(rms > np.percentile(rms, 40)) & (y > 76) & (y < 399)]
    if len(f0) < 20:
        return None
    st = 12 * np.log2(f0 / np.median(f0))
    st = st[np.abs(st) < 12]  # drop octave errors
    return float(st.std()) if len(st) > 10 else None


def compute(
    words: list[Word], audio: np.ndarray | None, audio_t0: float, question_t: float | None, sr: int = 16000
) -> AnswerFeatures:
    n = len(words)
    dur = (words[-1].end - words[0].start) if n else 0.0
    gaps = np.array([b.start - a.end for a, b in itertools.pairwise(words)]) if n > 1 else np.zeros(0)
    gaps = np.clip(gaps, 0, None)
    fill, reps = filler_count(words)
    # local speaking rate in 2 s bins
    rates = []
    if n > 4 and dur > 4:
        edges = np.arange(words[0].start, words[-1].end, 2.0)
        mids = np.array([(w.start + w.end) / 2 for w in words])
        counts = np.histogram(mids, bins=edges)[0] if len(edges) > 2 else np.array([])
        rates = counts[counts > 0].tolist()
    pitch_std, energy_cv = None, 0.0
    if audio is not None and n:
        a0 = max(0, int((words[0].start - audio_t0) * sr))
        a1 = min(len(audio), int((words[-1].end - audio_t0) * sr))
        seg = audio[a0:a1]
        pitch_std = pitch_semitone_std(seg, sr)
        word_rms = []
        for w in words:
            s0, s1 = int((w.start - audio_t0) * sr), int((w.end - audio_t0) * sr)
            if s1 > s0 and s0 >= 0 and s1 <= len(audio):
                word_rms.append(float(np.sqrt(np.mean(audio[s0:s1] ** 2)) + 1e-9))
        if len(word_rms) > 3:
            energy_cv = float(np.std(word_rms) / np.mean(word_rms))
    return AnswerFeatures(
        n_words=n,
        duration_s=float(dur),
        speech_rate_wpm=float(n / dur * 60) if dur > 0 else 0.0,
        filler_per_100w=100.0 * fill / max(n, 1),
        repair_per_100w=100.0 * reps / max(n, 1),
        gap_cv=float(gaps.std() / (gaps.mean() + 1e-3)) if len(gaps) > 2 else 0.0,
        long_pause_per_min=float((gaps > 0.6).sum() / max(dur, 1e-3) * 60) if len(gaps) else 0.0,
        rate_cv=float(np.std(rates) / (np.mean(rates) + 1e-6)) if len(rates) > 2 else 0.0,
        pitch_st_std=pitch_std,
        energy_cv=energy_cv,
        response_latency_s=(words[0].start - question_t) if (question_t is not None and n) else None,
        mean_word_prob=float(np.mean([w.prob for w in words])) if n else 0.0,
    )


def scripted_prior_score(f: AnswerFeatures, ai_text_prob: float | None) -> float:
    """Hand-set logistic prior (documented in docs/MODULES.md); replaced by the learned model in
    transcript/train.py once labelled answers from the custom test set exist."""
    z = -1.0
    z += 1.2 * (f.filler_per_100w < 1.0) - 0.08 * min(f.filler_per_100w, 10)
    z += 0.8 * (f.repair_per_100w < 0.5)
    z += 0.9 * (f.gap_cv < 0.9) - 0.3 * min(max(f.gap_cv - 1.2, 0), 2)
    z += 0.6 * (f.rate_cv < 0.25) if f.rate_cv else 0.0
    if f.pitch_st_std is not None:
        z += 0.8 * (f.pitch_st_std < 2.0) - 0.3 * (f.pitch_st_std > 3.5)
    if ai_text_prob is not None:
        z += 2.0 * (ai_text_prob - 0.5)
    if f.response_latency_s is not None:
        z += 0.5 * (f.response_latency_s > 6.0)
    return float(1 / (1 + np.exp(-z)))
