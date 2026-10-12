"""AV-sync evaluation.

1. Controlled desync study on genuine talking-head videos (any list: the SyncNet example clip, your own
   recordings, LRS/VoxCeleb clips): shift the audio by k frames (k in --shifts) and measure
     - offset accuracy: |estimated - true| <= 1 frame (40 ms)
     - detection: AUC / EER of the module risk for "desynced (|k| >= 4)" vs "in sync (k = 0)" windows.
2. LAV-DF / custom labelled videos: window-level AUC of the AV-sync risk against manipulation labels
   (run through the evaluation harness).

    python -m trusthire.avsync.eval --videos a.mp4 b.mp4 [--shifts -10 -6 -4 0 4 6 10]
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from trusthire.avsync.infer import SyncNetScorer, sync_risk
from trusthire.common.face import FaceStack, syncnet_crop
from trusthire.common.media import iter_frames, load_audio
from trusthire.common.registry import ensure
from trusthire.common.settings import REPO_ROOT
from trusthire.evaluation.metrics import report


def desync_study(videos: list[str], shifts: list[int], win: int = 75, hop: int = 37) -> dict:
    fs = FaceStack.get()
    sc = SyncNetScorer(vshift=12, max_stacks=12)
    rows: list[dict[str, Any]] = []
    for v in videos:
        frames = [f for _, f in iter_frames(v, 25)]
        audio = load_audio(v)
        faces = [fs.detector.detect(f, max_num=1) for f in frames[::5]]
        crops = []
        for i, f in enumerate(frames):
            fc = faces[min(i // 5, len(faces) - 1)]
            crops.append(syncnet_crop(f, fc[0]) if fc else np.zeros((224, 224, 3), np.uint8))
        # each video's own natural offset (capture latency) is the reference "0"
        base, _, _ = sc.offsets(crops, audio[: len(crops) * 640])
        for k in shifts:
            a = np.roll(audio, k * 640)
            for s in range(0, len(frames) - win, hop):
                off, conf, _ = sc.offsets(crops[s : s + win], a[s * 640 : (s + win) * 640])
                rows.append(
                    {
                        "video": v,
                        "shift": k,
                        "start": s,
                        "offset": int(off),
                        "conf": conf,
                        "expected": int(base - k),
                        "risk": sync_risk(off - base, conf),
                    }
                )
    res: dict = {"videos": videos, "n_windows": len(rows), "per_shift": {}}
    for k in shifts:
        rk = [r for r in rows if r["shift"] == k]
        res["per_shift"][k] = {
            "offset_acc_1f": float(np.mean([abs(r["offset"] - r["expected"]) <= 1 for r in rk])),
            "mean_conf": float(np.mean([r["conf"] for r in rk])),
            "mean_risk": float(np.mean([r["risk"] for r in rk])),
        }
    det = [r for r in rows if r["shift"] == 0 or abs(r["shift"]) >= 4]
    if det:
        y = np.array([int(abs(r["shift"]) >= 4) for r in det])
        if len(set(y)) == 2:
            res["detect_desync_ge4"] = report(y, np.array([r["risk"] for r in det]), bootstrap=300).as_dict()
    out = REPO_ROOT / "results" / "avsync"
    out.mkdir(parents=True, exist_ok=True)
    (out / "desync_study.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["per_shift"], indent=2))
    if "detect_desync_ge4" in res:
        d = res["detect_desync_ge4"]
        print(f"detect |shift|>=4 frames: AUC {d['auc']:.3f} EER {100 * d['eer']:.1f}% (n={d['n']})")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--shifts", nargs="*", type=int, default=[-10, -6, -4, -2, 0, 2, 4, 6, 10])
    a = ap.parse_args(argv)
    desync_study(a.videos or [str(ensure("avsync.example"))], a.shifts)


if __name__ == "__main__":
    main()
