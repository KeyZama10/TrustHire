"""`trusthire` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    ap = argparse.ArgumentParser(prog="trusthire")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download-models", help="download pretrained weights")
    d.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")
    d.add_argument("--only", nargs="*")

    sv = sub.add_parser("serve", help="run the API (+ built frontend)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")

    an = sub.add_parser("analyze", help="analyse a recorded interview offline")
    an.add_argument("video")
    an.add_argument("--id-photo")
    an.add_argument("--voice-sample")
    an.add_argument("--resume")
    an.add_argument("--report", action="store_true", help="also write a PDF report")
    an.add_argument("--candidate", default="Offline analysis")

    sub.add_parser("export-onnx", help="export/verify all ONNX models")
    sub.add_parser("bench-latency", help="per-module latency benchmark on the SyncNet example clip")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if a.cmd == "download-models":
        from trusthire.common.registry import main as dl

        return dl(["--profile", a.profile, *(["--only", *a.only] if a.only else [])])

    if a.cmd == "serve":
        import uvicorn

        uvicorn.run("trusthire.api.app:app", host=a.host, port=a.port, reload=a.reload, log_level="info")
        return 0

    if a.cmd == "analyze":
        return _analyze(a)

    if a.cmd == "export-onnx":
        from trusthire.audio.infer import AASISTBackend
        from trusthire.avsync.infer import ensure_onnx as sync_onnx
        from trusthire.video.infer import ensure_onnx as video_onnx

        for bb in ("effnb4", "xception"):
            print(video_onnx(bb))
        AASISTBackend("aasist")
        AASISTBackend("aasist_l")
        print(sync_onnx())
        return 0

    if a.cmd == "bench-latency":
        from trusthire.evaluation.latency import main as lat

        return lat()
    return 1


def _analyze(a: argparse.Namespace) -> int:
    import cv2

    from trusthire.common.media import load_audio
    from trusthire.common.settings import get_settings
    from trusthire.identity.infer import Enrollment, enroll_face_from_image, enroll_voice_from_audio
    from trusthire.pipeline.session import SessionStore
    from trusthire.transcript.nlp import read_resume
    from trusthire.worker.jobs import run_offline_job

    store = SessionStore(get_settings().sessions_dir)
    s = store.create(a.candidate, source="cli")
    enr = Enrollment()
    if a.id_photo:
        import numpy as np

        img = cv2.imread(a.id_photo)
        if img is None:
            raise SystemExit(f"cannot read image {a.id_photo}")
        enr.face = enroll_face_from_image(img)
        np.save(s.dir / "enroll_face.npy", enr.face)
    if a.voice_sample:
        import numpy as np

        enr.voice = enroll_voice_from_audio(load_audio(a.voice_sample))
        np.save(s.dir / "enroll_voice.npy", enr.voice)
    if a.resume:
        (s.dir / "resume.txt").write_text(read_resume(a.resume), encoding="utf-8")
    res = run_offline_job(s.id, a.video)
    print(json.dumps(res, indent=2))
    if a.report and res.get("state") == "done":
        s2 = store.get(s.id)
        s2 = type(s2).load(store.root, s.id)
        print("report:", s2.build_report())
    print("session dir:", s.dir)
    return 0 if res.get("state") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
