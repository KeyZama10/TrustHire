"""Offline-upload jobs. Celery + Redis when configured and reachable; otherwise an in-process thread pool
with the same interface (so `make demo` works on a laptop without Redis)."""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from trusthire.common.media import load_audio
from trusthire.common.settings import get_settings
from trusthire.fusion.live import alerts_from_point
from trusthire.pipeline.session import Session, point_to_dict

log = logging.getLogger(__name__)


def _job_file(session_dir: Path) -> Path:
    return session_dir / "job.json"


def write_status(session_dir: Path, **kw: Any) -> None:
    f = _job_file(session_dir)
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    d.update(kw)
    f.write_text(json.dumps(d), encoding="utf-8")


def read_status(session_dir: Path) -> dict[str, Any]:
    f = _job_file(session_dir)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"state": "unknown"}


def run_offline_job(session_id: str, video_path: str) -> dict[str, Any]:
    """The actual work (runs in a Celery worker or a local thread)."""
    from trusthire.pipeline.offline import analyze_file

    s = get_settings()
    sess = Session.load(s.sessions_dir, session_id)
    write_status(sess.dir, state="running", progress=0.0)
    try:
        res = analyze_file(
            video_path,
            sess.enrollment,
            sess.resume_text,
            progress=lambda p: write_status(sess.dir, state="running", progress=round(p, 3)),
        )
        sess.points = [point_to_dict(p) for p in res.points]
        sess.alerts = []
        for p in res.points:
            for a in alerts_from_point(p):
                ad = a.model_dump()
                if not any(
                    x["module"] == ad["module"] and abs(x["t"] - ad["t"]) < 3.1 for x in sess.alerts[-6:]
                ):
                    sess.alerts.append(ad)
        sess.answers = [{k: v for k, v in a.items()} for a in res.answers]
        sf.write(sess.dir / "audio.wav", load_audio(video_path), 16000)
        sess.save_flagged(res.flagged_frames)
        sess.meta.update(
            {
                "status": "done",
                "duration_s": res.duration_s,
                "wall_s": round(res.wall_s, 1),
                "latency_ms_p50": float(np.median(res.latency_ms)) if res.latency_ms else None,
            }
        )
        sess.save()
        write_status(sess.dir, state="done", progress=1.0)
        return {"state": "done", "trust": sess.meta.get("trust")}
    except Exception as e:
        log.exception("offline job failed")
        sess.meta["status"] = "failed"
        sess.save()
        write_status(sess.dir, state="failed", error=f"{e}\n{traceback.format_exc()[-1500:]}")
        return {"state": "failed", "error": str(e)}


class JobRunner:
    _local = ThreadPoolExecutor(max_workers=1, thread_name_prefix="offline-job")
    _lock = threading.Lock()

    @staticmethod
    def celery_available() -> bool:
        s = get_settings()
        if not s.use_celery:
            return False
        try:
            import redis

            redis.Redis.from_url(s.redis_url, socket_connect_timeout=1).ping()
            return True
        except Exception:
            return False

    @classmethod
    def submit(cls, session: Session, video_path: Path) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:10]
        write_status(session.dir, state="queued", progress=0.0, job_id=job_id)
        if cls.celery_available():
            from trusthire.worker.celery_app import analyze_upload

            analyze_upload.delay(session.id, str(video_path))
            backend = "celery"
        else:
            cls._local.submit(run_offline_job, session.id, str(video_path))
            backend = "local-thread"
        write_status(session.dir, backend=backend)
        return {"job_id": job_id, "session_id": session.id, "backend": backend}
