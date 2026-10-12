"""Transcript branch: answer segmentation (VAD) -> Whisper -> scripted-answer + resume-consistency scores.

ASR is far slower than the 1.5 s hop on CPU, so this branch is asynchronous: utterances are analysed in a
background thread and results are delivered through `on_result`, which the session uses to update the
already-emitted timeline points that overlap the utterance.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from trusthire.common.types import Modality, ModuleOutput
from trusthire.transcript import features as feat

if TYPE_CHECKING:
    from trusthire.transcript.nlp import ResumeIndex

log = logging.getLogger(__name__)
SR = 16000


@dataclass
class AnswerResult:
    t_start: float
    t_end: float
    text: str
    features: dict
    ai_text_prob: float | None
    scripted: ModuleOutput
    resume: ModuleOutput | None
    words: list[dict] = field(default_factory=list)


class TranscriptMonitor:
    def __init__(
        self,
        whisper_model: str = "small",
        compute_type: str = "int8",
        resume_text: str | None = None,
        on_result: Callable[[AnswerResult], None] | None = None,
        min_silence_s: float = 1.2,
        max_utt_s: float = 30.0,
        use_ai_text: bool = True,
        synchronous: bool = False,
    ):
        self.whisper_model = whisper_model
        self.compute_type = compute_type
        self.on_result = on_result
        self.min_silence_s = min_silence_s
        self.max_utt_s = max_utt_s
        self.use_ai_text = use_ai_text
        self.synchronous = synchronous
        self.buf = np.zeros(0, np.float32)
        self.buf_t0 = 0.0
        self.questions: list[float] = []
        self.results: list[AnswerResult] = []
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self._pending: list[Future] = []
        self._lock = threading.Lock()
        self.resume_index: ResumeIndex | None = None
        if resume_text:
            self.set_resume(resume_text)

    def set_resume(self, text: str) -> None:
        from trusthire.transcript.nlp import ResumeIndex

        self.resume_index = ResumeIndex(text)

    def mark_question(self, t: float) -> None:
        self.questions.append(t)

    # ------------------------------------------------------------------ streaming
    def push(self, samples: np.ndarray, t0: float) -> None:
        """Append contiguous audio starting at stream time t0."""
        with self._lock:
            if self.buf.size == 0:
                self.buf_t0 = t0
            self.buf = np.concatenate([self.buf, samples.astype(np.float32)])
            self._segment(final=False)

    def flush(self, wait: bool = True) -> list[AnswerResult]:
        with self._lock:
            self._segment(final=True)
        if wait:
            for f in list(self._pending):
                f.result()
        return self.results

    def _segment(self, final: bool) -> None:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        if self.buf.size < SR:
            return
        segs = get_speech_timestamps(
            self.buf,
            VadOptions(min_silence_duration_ms=int(self.min_silence_s * 1000), speech_pad_ms=150),
            sampling_rate=SR,
        )
        if not segs:
            # drop old silence but keep 2 s of context
            keep = 2 * SR
            if self.buf.size > keep:
                self.buf_t0 += (self.buf.size - keep) / SR
                self.buf = self.buf[-keep:]
            return
        first, last = segs[0], segs[-1]
        tail_silence = (self.buf.size - last["end"]) / SR
        span = (last["end"] - first["start"]) / SR
        if final or (tail_silence >= self.min_silence_s and span >= 2.0) or span >= self.max_utt_s:
            a0, a1 = max(0, first["start"] - SR // 4), min(self.buf.size, last["end"] + SR // 4)
            audio = self.buf[a0:a1].copy()
            t_audio0 = self.buf_t0 + a0 / SR
            self.buf_t0 += a1 / SR
            self.buf = self.buf[a1:]
            if len(audio) >= SR:
                fut = self._pool.submit(self._analyse, audio, t_audio0)
                self._pending.append(fut)
                if self.synchronous:
                    fut.result()

    # ------------------------------------------------------------------ analysis
    def _analyse(self, audio: np.ndarray, t0: float) -> AnswerResult | None:
        from trusthire.transcript.asr import ASR

        try:
            tic = time.perf_counter()
            tr = ASR.get(self.whisper_model, self.compute_type).transcribe(audio, offset_s=t0)
            if not tr.words:
                return None
            q = max((qt for qt in self.questions if qt <= tr.words[0].start + 0.5), default=None)
            q = q if q is not None and tr.words[0].start - q < 90 else None
            f = feat.compute(tr.words, audio, t0, q)
            ai_p = None
            if self.use_ai_text:
                from trusthire.transcript.nlp import AITextDetector

                ai_p = AITextDetector.get().prob_ai(tr.text)
            score = feat.scripted_prior_score(f, ai_p)
            conf = float(np.clip(f.n_words / 40.0, 0.1, 1.0)) * 0.8
            ts, te = tr.words[0].start, tr.words[-1].end
            lat = (time.perf_counter() - tic) * 1e3
            scripted = ModuleOutput(
                module=Modality.SCRIPTED,
                score=score,
                confidence=conf,
                t_start=ts,
                t_end=te,
                latency_ms=lat,
                details={
                    **f.as_dict(),
                    "ai_text_prob": ai_p,
                    "text": tr.text[:2000],
                    "language": tr.language,
                },
            )
            resume_out = None
            if self.resume_index is not None:
                res = self.resume_index.check(tr.text)
                if res is not None:
                    resume_out = ModuleOutput(
                        module=Modality.RESUME,
                        score=res.score,
                        confidence=float(np.clip(res.relevance, 0, 1)) * 0.8,
                        t_start=ts,
                        t_end=te,
                        details={
                            "support": round(res.support, 3),
                            "relevance": round(res.relevance, 3),
                            "checks": [c.__dict__ for c in res.checks],
                        },
                    )
            r = AnswerResult(
                ts,
                te,
                tr.text,
                f.as_dict(),
                ai_p,
                scripted,
                resume_out,
                words=[{"t": round(w.start, 2), "w": w.text} for w in tr.words],
            )
            self.results.append(r)
            if self.on_result:
                self.on_result(r)
            return r
        except Exception:
            log.exception("transcript analysis failed")
            return None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
