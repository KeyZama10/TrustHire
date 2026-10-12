"""Celery application for offline interview analysis.

celery -A trusthire.worker.celery_app worker --loglevel=info --concurrency=1 [-P solo on Windows]
"""

from __future__ import annotations

from typing import Any

from celery import Celery

from trusthire.common.settings import get_settings

_s = get_settings()
app = Celery("trusthire", broker=_s.redis_url, backend=_s.redis_url)
app.conf.update(
    task_acks_late=True, worker_prefetch_multiplier=1, task_track_started=True, result_expires=7 * 24 * 3600
)


@app.task(name="trusthire.analyze_upload")
def analyze_upload(session_id: str, video_path: str) -> dict[str, Any]:
    from trusthire.worker.jobs import run_offline_job

    return run_offline_job(session_id, video_path)
