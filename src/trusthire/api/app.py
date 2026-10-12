"""TrustHire API: REST + WebSocket + WebRTC signalling. Serves the built React app from frontend/dist."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from trusthire import __version__
from trusthire.common.settings import REPO_ROOT, get_settings
from trusthire.pipeline.session import SessionStore

log = logging.getLogger("trusthire.api")
STORE = SessionStore()
STARTED = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    # warm the shared models in the background so the first session starts quickly
    async def warm() -> None:
        from trusthire.pipeline.analyzer import SharedModels

        try:
            await asyncio.get_running_loop().run_in_executor(None, SharedModels.get)
            app.state.models_ready = True
        except Exception:
            log.exception("model warm-up failed")

    app.state.models_ready = False
    task = (
        asyncio.create_task(warm())
        if not os.environ.get("TRUSTHIRE_SKIP_WARMUP")
        else asyncio.create_task(asyncio.sleep(0))
    )
    yield
    task.cancel()
    from trusthire.api.webrtc import PEERS, close_peer

    for sid in list(PEERS):
        await close_peer(sid)


app = FastAPI(title="TrustHire", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSession(BaseModel):
    candidate: str


class Offer(BaseModel):
    sdp: str
    type: str


class Mark(BaseModel):
    t: float | None = None


def _session(sid: str):  # type: ignore[no-untyped-def]
    try:
        return STORE.get(sid)
    except KeyError:
        raise HTTPException(404, "session not found") from None


# ------------------------------------------------------------------ system
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": __version__,
        "models_ready": app.state.models_ready,
        "uptime_s": round(time.time() - STARTED),
    }


@app.get("/api/system")
def system() -> dict[str, Any]:
    import onnxruntime as ort
    import torch

    from trusthire.worker.jobs import JobRunner

    s = get_settings()
    return {
        "profile": s.profile,
        "onnx_providers": ort.get_available_providers(),
        "cuda": torch.cuda.is_available(),
        "models": {
            "video": s.video_backbone,
            "audio": s.audio_model,
            "avsync": "syncnet",
            "face": "arcface (buffalo_l)",
            "speaker": "ecapa-tdnn",
            "landmarks": "mediapipe face landmarker",
            "objects": s.yolo_model,
            "asr": f"faster-whisper {s.whisper_model}",
            "fusion": s.fusion_method,
        },
        "llm_provider": s.llm_provider,
        "celery": JobRunner.celery_available(),
        "models_ready": app.state.models_ready,
        "window_s": s.window_s,
        "hop_s": s.hop_s,
    }


# ------------------------------------------------------------------ sessions
@app.post("/api/sessions")
def create_session(body: CreateSession) -> dict[str, Any]:
    return STORE.create(body.candidate.strip() or "Unnamed candidate").summary()


@app.get("/api/sessions")
def list_sessions() -> list[dict[str, Any]]:
    return STORE.list()


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    from trusthire.pipeline.session import module_stats, session_trust_from_dicts
    from trusthire.worker.jobs import read_status

    s = _session(sid)
    frames_idx = s.dir / "frames" / "index.json"
    return {
        **s.summary(),
        "points": s.points,
        "alerts": s.alerts,
        "answers": s.answers,
        "trust": session_trust_from_dicts(s.points),
        "module_stats": module_stats(s.points),
        "job": read_status(s.dir),
        "frames": json.loads(frames_idx.read_text()) if frames_idx.exists() else [],
        "has_report": s.report_path().exists(),
        "dropped_windows": s.dropped,
    }


@app.delete("/api/sessions/{sid}")
def delete_session(sid: str) -> dict[str, bool]:
    s = _session(sid)
    if s.analyzer is not None:
        raise HTTPException(409, "session is live")
    STORE.live.pop(sid, None)
    shutil.rmtree(s.dir, ignore_errors=True)
    return {"deleted": True}


@app.post("/api/sessions/{sid}/enroll/face")
async def enroll_face(sid: str, file: UploadFile = File(...)) -> dict[str, Any]:
    from trusthire.identity.infer import enroll_face_from_image

    s = _session(sid)
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    try:
        emb = await asyncio.get_running_loop().run_in_executor(None, enroll_face_from_image, img)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    np.save(s.dir / "enroll_face.npy", emb)
    cv2.imwrite(str(s.dir / "enroll_photo.jpg"), img)
    s.enrollment.face, s.enrollment.face_source = emb, "ID photo"
    return {"ok": True, "enrolled_face": True}


@app.post("/api/sessions/{sid}/enroll/voice")
async def enroll_voice(sid: str, file: UploadFile = File(...)) -> dict[str, Any]:
    from trusthire.common.media import load_audio
    from trusthire.identity.infer import enroll_voice_from_audio

    s = _session(sid)
    suffix = Path(file.filename or "voice.webm").suffix or ".webm"
    p = s.dir / f"enroll_voice{suffix}"
    p.write_bytes(await file.read())
    try:
        audio = await asyncio.get_running_loop().run_in_executor(None, load_audio, p)
        emb = await asyncio.get_running_loop().run_in_executor(None, enroll_voice_from_audio, audio)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    np.save(s.dir / "enroll_voice.npy", emb)
    s.enrollment.voice, s.enrollment.voice_source = emb, "voice sample"
    return {"ok": True, "enrolled_voice": True, "seconds": round(len(audio) / 16000, 1)}


@app.post("/api/sessions/{sid}/resume")
async def upload_resume(sid: str, file: UploadFile = File(...)) -> dict[str, Any]:
    from trusthire.transcript.nlp import chunk_resume, read_resume

    s = _session(sid)
    p = s.dir / f"resume{Path(file.filename or 'resume.txt').suffix}"
    p.write_bytes(await file.read())
    text = read_resume(p)
    if not text.strip():
        raise HTTPException(422, "no text could be extracted from the resume")
    (s.dir / "resume.txt").write_text(text, encoding="utf-8")
    s.resume_text = text
    return {"ok": True, "chunks": len(chunk_resume(text))}


# ------------------------------------------------------------------ live
@app.post("/api/sessions/{sid}/webrtc/offer")
async def webrtc_offer(sid: str, offer: Offer) -> dict[str, str]:
    from trusthire.api.webrtc import handle_offer

    s = _session(sid)
    if s.meta.get("status") == "done":
        raise HTTPException(409, "session already finished")
    return await handle_offer(s, offer.sdp, offer.type)


@app.post("/api/sessions/{sid}/question")
def mark_question(sid: str, body: Mark) -> dict[str, Any]:
    s = _session(sid)
    if s.analyzer is None:
        raise HTTPException(409, "session is not live")
    t = body.t if body.t is not None else (s.points[-1]["t_end"] if s.points else 0.0)
    s.analyzer.mark_question(t)
    return {"ok": True, "t": t}


@app.post("/api/sessions/{sid}/stop")
async def stop(sid: str) -> dict[str, Any]:
    from trusthire.api.webrtc import close_peer

    s = _session(sid)
    await close_peer(sid)
    await asyncio.get_running_loop().run_in_executor(None, s.stop)
    return s.summary()


@app.websocket("/ws/sessions/{sid}")
async def ws(websocket: WebSocket, sid: str) -> None:
    await websocket.accept()
    try:
        s = STORE.get(sid)
    except KeyError:
        await websocket.close(code=4404)
        return
    s.loop = asyncio.get_running_loop()
    q: asyncio.Queue[str] = asyncio.Queue()
    s.subscribers.add(q)
    try:
        await websocket.send_json(
            {"type": "hello", "session": s.summary(), "points": s.points[-200:], "alerts": s.alerts}
        )
        while True:
            msg = await q.get()
            await websocket.send_text(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        s.subscribers.discard(q)


# ------------------------------------------------------------------ offline + reports
@app.post("/api/uploads")
async def upload_interview(
    video: UploadFile = File(...),
    candidate: str = Form("Uploaded interview"),
    id_photo: UploadFile | None = File(None),
    resume: UploadFile | None = File(None),
) -> dict[str, Any]:
    from trusthire.worker.jobs import JobRunner

    s = STORE.create(candidate, source="upload")
    vp = s.dir / f"input{Path(video.filename or 'interview.mp4').suffix or '.mp4'}"
    with open(vp, "wb") as f:
        while chunk := await video.read(1 << 20):
            f.write(chunk)
    if id_photo is not None and id_photo.filename:
        await enroll_face(s.id, id_photo)
    if resume is not None and resume.filename:
        await upload_resume(s.id, resume)
    s.meta["status"] = "processing"
    s.save()
    return JobRunner.submit(s, vp)


@app.get("/api/sessions/{sid}/job")
def job_status(sid: str) -> dict[str, Any]:
    from trusthire.worker.jobs import read_status

    return read_status(_session(sid).dir)


@app.post("/api/sessions/{sid}/report")
async def make_report(sid: str) -> dict[str, Any]:
    s = _session(sid)
    t = time.perf_counter()
    await asyncio.get_running_loop().run_in_executor(None, s.build_report)
    return {
        "ok": True,
        "seconds": round(time.perf_counter() - t, 1),
        "summary": (s.dir / "summary.txt").read_text(encoding="utf-8"),
    }


@app.get("/api/sessions/{sid}/report.pdf")
async def get_report(sid: str) -> FileResponse:
    s = _session(sid)
    if not s.report_path().exists():
        await asyncio.get_running_loop().run_in_executor(None, s.build_report)
    return FileResponse(s.report_path(), media_type="application/pdf", filename=f"trusthire-{sid}.pdf")


@app.get("/api/sessions/{sid}/frames/{name}")
def get_frame(sid: str, name: str) -> FileResponse:
    s = _session(sid)
    p = s.dir / "frames" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/sessions/{sid}/summary")
def get_summary(sid: str) -> JSONResponse:
    p = _session(sid).dir / "summary.txt"
    return JSONResponse({"summary": p.read_text(encoding="utf-8") if p.exists() else None})


# ------------------------------------------------------------------ frontend
DIST = REPO_ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
