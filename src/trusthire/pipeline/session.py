"""Session store: persistence, live ingest worker, WebSocket fan-out and report generation."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import soundfile as sf

from trusthire.common.settings import get_settings
from trusthire.common.types import FusedPoint, Window
from trusthire.fusion.live import alerts_from_point, session_trust
from trusthire.identity.infer import Enrollment

log = logging.getLogger(__name__)


def point_to_dict(p: FusedPoint) -> dict[str, Any]:
    return {
        "t": round(p.t, 3),
        "t_start": round(p.t_start, 3),
        "t_end": round(p.t_end, 3),
        "risk": round(p.risk, 4),
        "latency_ms": None if p.latency_ms is None else round(p.latency_ms, 1),
        "alerts": p.alerts,
        "contributions": p.contributions,
        "modules": {
            k: {
                "score": round(o.score, 4),
                "confidence": round(o.confidence, 3),
                "missing": bool(o.details.get("missing")),
                "latency_ms": round(o.latency_ms, 1),
                "details": _jsonable(o.details),
            }
            for k, o in p.modules.items()
        },
    }


def _jsonable(d: Any) -> Any:
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, float)):
        return None if not np.isfinite(d) else round(float(d), 5)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


class Session:
    def __init__(self, sid: str, root: Path, meta: dict[str, Any] | None = None):
        self.id = sid
        self.dir = root / sid
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = meta or {
            "id": sid,
            "candidate": "",
            "status": "created",
            "created": datetime.now(UTC).isoformat(),
            "source": "live",
        }
        self.points: list[dict[str, Any]] = []
        self.alerts: list[dict[str, Any]] = []
        self.answers: list[dict[str, Any]] = []
        self.enrollment = Enrollment()
        self.resume_text: str | None = None
        self.subscribers: set[asyncio.Queue[str]] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.analyzer: Any = None
        self._q: queue.Queue[Window | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self.dropped = 0
        self._audio_chunks: list[np.ndarray] = []
        self._flagged: list[tuple[float, np.ndarray, float]] = []
        self.t0: float | None = None
        self._load_enrollment()

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        trust = session_trust_from_dicts(self.points)
        self.meta.update(
            {
                "n_windows": len(self.points),
                "trust": trust,
                "n_alerts": len(self.alerts),
                "duration_s": self.points[-1]["t_end"] if self.points else self.meta.get("duration_s", 0),
                "dropped_windows": self.dropped,
            }
        )
        data = {
            **self.meta,
            "points": self.points,
            "alerts": self.alerts,
            "answers": self.answers,
            "module_stats": module_stats(self.points),
        }
        tmp = self.dir / "session.json.tmp"
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.dir / "session.json")

    @classmethod
    def load(cls, root: Path, sid: str) -> Session:
        d = json.loads((root / sid / "session.json").read_text(encoding="utf-8"))
        s = cls(
            sid,
            root,
            {k: v for k, v in d.items() if k not in ("points", "alerts", "answers", "module_stats")},
        )
        s.points, s.alerts, s.answers = d.get("points", []), d.get("alerts", []), d.get("answers", [])
        rt = root / sid / "resume.txt"
        s.resume_text = rt.read_text(encoding="utf-8") if rt.exists() else None
        return s

    def _load_enrollment(self) -> None:
        f, v = self.dir / "enroll_face.npy", self.dir / "enroll_voice.npy"
        if f.exists():
            self.enrollment.face = np.load(f)
            self.enrollment.face_source = "ID photo"
        if v.exists():
            self.enrollment.voice = np.load(v)
            self.enrollment.voice_source = "voice sample"

    def summary(self) -> dict[str, Any]:
        return {
            k: self.meta.get(k)
            for k in (
                "id",
                "candidate",
                "status",
                "created",
                "source",
                "trust",
                "n_windows",
                "n_alerts",
                "duration_s",
            )
        } | {
            "enrolled_face": self.enrollment.face is not None,
            "enrolled_voice": self.enrollment.voice is not None,
            "has_resume": self.resume_text is not None,
        }

    # ------------------------------------------------------------------ fan-out
    def broadcast(self, msg: dict[str, Any]) -> None:
        if not self.loop:
            return
        text = json.dumps(msg)
        for q in list(self.subscribers):
            self.loop.call_soon_threadsafe(q.put_nowait, text)

    def add_point(self, p: FusedPoint, index: int | None = None) -> None:
        d = point_to_dict(p)
        if index is None:
            self.points.append(d)
            for a in alerts_from_point(p):
                ad = a.model_dump()
                # de-duplicate consecutive identical alerts (overlapping windows)
                if not any(
                    x["module"] == ad["module"] and abs(x["t"] - ad["t"]) < 3.1 for x in self.alerts[-6:]
                ):
                    self.alerts.append(ad)
                    self.broadcast({"type": "alert", "alert": ad})
            self.broadcast({"type": "point", "point": d, "trust": session_trust_from_dicts(self.points)})
        else:
            self.points[index] = d
            self.broadcast({"type": "update", "index": index, "point": d})

    # ------------------------------------------------------------------ live analysis
    def start_analysis(self, loop: asyncio.AbstractEventLoop, source: str = "live") -> None:
        from trusthire.pipeline.analyzer import SessionAnalyzer

        self.loop = loop
        self.meta["status"] = "live"
        self.meta["source"] = source
        self.analyzer = SessionAnalyzer(
            self.enrollment,
            self.resume_text,
            on_point=lambda p: self.add_point(p),
            on_update=lambda i, p: self.add_point(p, i),
            on_transcript=lambda r: self._on_answer(r),
        )
        self._worker = threading.Thread(target=self._run, daemon=True, name=f"session-{self.id}")
        self._worker.start()
        self.broadcast({"type": "status", "status": "live"})

    def submit(self, w: Window) -> None:
        # keep the live view current: if analysis falls behind by >2 windows, drop the oldest
        while self._q.qsize() > 2:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                break
        self._audio_chunks.append(
            w.audio[-int(get_settings().hop_s * w.sample_rate) :] if self._audio_chunks else w.audio
        )
        self._q.put(w)

    def _run(self) -> None:
        while True:
            w = self._q.get()
            if w is None:
                break
            try:
                self.analyzer.process_window(w)
                for idx, crop, prob in w.cache.get("video_crops", []):
                    self._flagged.append((w.frame_times[idx] - (self.t0 or 0), crop, prob))
                self._flagged = sorted(self._flagged, key=lambda x: -x[2])[:12]
            except Exception:
                log.exception("window %s failed", w.index)

    def _on_answer(self, r: Any) -> None:
        a = {
            "t_start": r.t_start,
            "t_end": r.t_end,
            "text": r.text,
            "features": _jsonable(r.features),
            "ai_text_prob": r.ai_text_prob,
            "scripted_score": r.scripted.score,
            "resume": None if r.resume is None else _jsonable({"score": r.resume.score, **r.resume.details}),
            "words": r.words,
        }
        self.answers.append(a)
        self.broadcast({"type": "answer", "answer": a})

    def stop(self) -> None:
        if self._worker is not None:
            self._q.put(None)
            self._worker.join(timeout=60)
        if self.analyzer is not None:
            self.analyzer.finish()
            self.analyzer.close()
            self.analyzer = None
        if self._audio_chunks:
            sf.write(self.dir / "audio.wav", np.concatenate(self._audio_chunks), 16000)
        self.save_flagged(self._flagged)
        self.meta["status"] = "done"
        self.save()
        self.broadcast({"type": "status", "status": "done", "trust": self.meta.get("trust")})

    def save_flagged(self, flagged: list[tuple[float, np.ndarray, float]]) -> None:
        keep: list[tuple[float, np.ndarray, float]] = []
        for f in sorted(flagged, key=lambda x: -x[2]):
            if all(abs(f[0] - k[0]) > 1.0 for k in keep):
                keep.append(f)
            if len(keep) >= 6:
                break
        fd = self.dir / "frames"
        fd.mkdir(exist_ok=True)
        index = []
        for i, (t, crop, p) in enumerate(keep):
            cv2.imwrite(str(fd / f"crop_{i}.jpg"), crop)
            index.append({"file": f"crop_{i}.jpg", "t": round(float(t), 2), "p_fake": round(float(p), 4)})
        (fd / "index.json").write_text(json.dumps(index), encoding="utf-8")

    # ------------------------------------------------------------------ report
    def report_path(self) -> Path:
        return self.dir / "report.pdf"

    def build_report(self) -> Path:
        from trusthire.explain import report, visual
        from trusthire.explain.llm import build_evidence, summarize

        s = get_settings()
        self.save()
        data = json.loads((self.dir / "session.json").read_text(encoding="utf-8"))
        figs: dict[str, bytes] = {}
        if self.points:
            figs["timeline"] = visual.timeline_figure(self.points, s.alert_risk_threshold)
            top = max(self.points, key=lambda p: p["risk"])
            if top.get("contributions"):
                figs["contributions"] = visual.contributions_figure(
                    top["contributions"], f"Fusion attribution at t={top['t']:.1f}s"
                )
        frames: list[tuple[float, bytes, float]] = []
        idx = self.dir / "frames" / "index.json"
        if idx.exists():
            for f in json.loads(idx.read_text(encoding="utf-8"))[:6]:
                crop = cv2.imread(str(self.dir / "frames" / f["file"]))
                if crop is None:
                    continue
                overlay, p = visual.gradcam(crop, s.video_backbone)
                cv2.imwrite(str(self.dir / "frames" / f"cam_{f['file']}"), overlay)
                _ok, png = cv2.imencode(".png", overlay)
                frames.append((f["t"], png.tobytes(), p))
        wav = self.dir / "audio.wav"
        audio_pts = [p for p in self.points if not p["modules"].get("voice_spoof", {}).get("missing", True)]
        if wav.exists() and audio_pts:
            from trusthire.pipeline.analyzer import SharedModels

            top = max(audio_pts, key=lambda p: p["modules"]["voice_spoof"]["score"])
            audio, sr = sf.read(wav, dtype="float32")
            t1 = int(top["t_end"] * sr)
            seg = audio[max(0, t1 - 64600) : t1]
            if len(seg) > sr:
                be = SharedModels.get().audio.backend
                sal, p = visual.audio_occlusion(seg, be.p_spoof)
                figs["audio_saliency"] = visual.saliency_figure(
                    seg, sal, p, f"t={top['t_start']:.1f}-{top['t_end']:.1f}s"
                )
        means = {m: v["mean"] for m, v in data.get("module_stats", {}).items()}
        ev = build_evidence(
            data.get("trust", {}), self.alerts, means, self.answers, data.get("duration_s", 0)
        )
        summary, provider = summarize(
            ev, s.llm_provider, s.llm_model if s.llm_provider == "anthropic" else s.ollama_model
        )
        data["fusion_method"] = s.fusion_method
        (self.dir / "summary.txt").write_text(summary, encoding="utf-8")
        return report.build_pdf(data, summary, provider, figs, frames, self.report_path())


def session_trust_from_dicts(points: list[dict[str, Any]]) -> dict[str, float]:
    class _P:
        def __init__(self, r: float):
            self.risk = r

    return session_trust([_P(p["risk"]) for p in points])  # type: ignore[misc]


def module_stats(points: list[dict[str, Any]], flag: float = 0.65) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if not points:
        return out
    mods = sorted({m for p in points for m in p["modules"]})
    for m in mods:
        vals = [
            p["modules"][m]["score"] for p in points if m in p["modules"] and not p["modules"][m]["missing"]
        ]
        out[m] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "max": float(np.max(vals)) if vals else 0.0,
            "coverage": len(vals) / len(points),
            "flagged": int(sum(v >= flag for v in vals)),
        }
    return out


class SessionStore:
    def __init__(self, root: Path | None = None):
        self.root = root or get_settings().sessions_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.live: dict[str, Session] = {}

    def create(self, candidate: str, source: str = "live") -> Session:
        sid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        s = Session(sid, self.root)
        s.meta.update({"candidate": candidate, "source": source})
        s.save()
        self.live[sid] = s
        return s

    def get(self, sid: str) -> Session:
        cached = self.live.get(sid)
        # only sessions being analysed live are authoritative in memory; anything else (e.g. results written
        # by an offline job in another thread/process) must be re-read from disk
        if cached is not None and (cached.analyzer is not None or cached.subscribers):
            return cached
        if not (self.root / sid / "session.json").exists():
            raise KeyError(sid)
        s = Session.load(self.root, sid)
        self.live[sid] = s
        return s

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*/session.json"), reverse=True):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                out.append(
                    {
                        k: d.get(k)
                        for k in (
                            "id",
                            "candidate",
                            "status",
                            "created",
                            "source",
                            "trust",
                            "n_windows",
                            "n_alerts",
                            "duration_s",
                        )
                    }
                )
            except Exception:
                continue
        return out
