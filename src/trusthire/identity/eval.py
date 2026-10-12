"""Identity-module evaluation.

Face: LFW verification pairs (test split, 500 genuine + 500 impostor) through the full TrustHire face path
(SCRFD detection -> 5-point alignment -> ArcFace). Reports AUC/EER, accuracy at the EER threshold, and the
FMR / FNMR at the configured `face_match_threshold`.
Speaker: LibriSpeech test-clean trials (same/different speaker, 4 s segments = live context) with ECAPA-TDNN.

    python -m trusthire.identity.eval face
    python -m trusthire.identity.eval speaker --trials 2000
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import cv2
import numpy as np

from trusthire.common.registry import data_dir
from trusthire.common.settings import REPO_ROOT, get_settings
from trusthire.evaluation.metrics import report

OUT = REPO_ROOT / "results" / "identity"


def _rates(y: np.ndarray, s: np.ndarray, thr: float) -> dict[str, float]:
    """y=1 means *different* person (the suspicious class); s = risk = 1 - similarity."""
    fnmr = float((s[y == 0] >= 1 - thr).mean())  # genuine pairs wrongly flagged
    fmr = float((s[y == 1] < 1 - thr).mean())  # impostors accepted
    return {"threshold_sim": thr, "false_flag_rate_same_person": fnmr, "missed_impostor_rate": fmr}


def face_lfw() -> dict[str, Any]:
    from sklearn.datasets import fetch_lfw_pairs

    from trusthire.common.face import FaceStack

    lfw = fetch_lfw_pairs(
        subset="test",
        color=True,
        resize=1.0,
        slice_=None,
        data_home=str(data_dir() / "public" / "lfw"),
        download_if_missing=True,
    )
    fs = FaceStack.get()
    sims, ys, no_face = [], [], 0
    for pair, target in zip(lfw.pairs, lfw.target, strict=True):
        embs = []
        for img in pair:
            bgr = cv2.cvtColor((img * (255 if img.max() <= 1 else 1)).astype(np.uint8), cv2.COLOR_RGB2BGR)
            # LFW images are 250x250 with the face centred; pad so the detector sees context at 320 px
            faces = fs.detector.detect(bgr, max_num=1)
            if not faces:
                no_face += 1
                break
            embs.append(fs.arcface.embed(bgr, faces[0]))
        if len(embs) < 2:
            continue
        sims.append(float(embs[0] @ embs[1]))
        ys.append(1 - int(target))  # sklearn: target 1 = same person -> our y=0
    s = 1 - np.array(sims)
    y = np.array(ys)
    r = report(y, s, probabilistic=False, bootstrap=500).as_dict()
    acc = float(((s >= r["eer_threshold"]) == y).mean())
    res = {
        "benchmark": "LFW test pairs",
        "n_pairs": len(y),
        "pairs_without_face": no_face,
        **r,
        "accuracy_at_eer_thr": acc,
        "eer_threshold_sim": 1 - r["eer_threshold"],
        "at_configured_threshold": _rates(y, s, get_settings().face_match_threshold),
        "mean_sim_same": float(np.mean(1 - s[y == 0])),
        "mean_sim_diff": float(np.mean(1 - s[y == 1])),
    }
    _save("face_lfw", res)
    return res


def speaker_librispeech(n_trials: int = 2000, seg_s: float = 4.04, seed: int = 0) -> dict[str, Any]:
    import soundfile as sf
    import torchaudio

    from trusthire.identity.speaker import SpeakerEncoder

    root = data_dir() / "public" / "librispeech"
    root.mkdir(parents=True, exist_ok=True)
    ds = torchaudio.datasets.LIBRISPEECH(str(root), url="test-clean", download=True)
    rng = np.random.default_rng(seed)
    base = root / "LibriSpeech" / "test-clean"
    by_spk: dict[str, list] = {}
    for p in base.rglob("*.flac"):
        by_spk.setdefault(p.parts[-3], []).append(p)
    spks = sorted(by_spk)
    enc = SpeakerEncoder.get()
    n = int(seg_s * 16000)
    cache: dict[str, np.ndarray] = {}

    def emb(p) -> np.ndarray:  # type: ignore[no-untyped-def]
        k = str(p)
        if k not in cache:
            x, _ = sf.read(p, dtype="float32")
            s0 = rng.integers(0, max(1, len(x) - n))
            cache[k] = enc.embed(x[s0 : s0 + n])
        return cache[k]

    ys, sims = [], []
    for i in range(n_trials):
        same = i % 2 == 0
        a = spks[rng.integers(len(spks))]
        b = a if same else spks[rng.integers(len(spks))]
        while not same and b == a:
            b = spks[rng.integers(len(spks))]
        fa = by_spk[a][rng.integers(len(by_spk[a]))]
        fb = by_spk[b][rng.integers(len(by_spk[b]))]
        while same and fb == fa and len(by_spk[a]) > 1:
            fb = by_spk[b][rng.integers(len(by_spk[b]))]
        sims.append(float(emb(fa) @ emb(fb)))
        ys.append(0 if same else 1)
    s, y = 1 - np.array(sims), np.array(ys)
    r = report(y, s, probabilistic=False, bootstrap=500).as_dict()
    res = {
        "benchmark": "LibriSpeech test-clean trials (4 s segments)",
        "n_trials": len(y),
        "n_speakers": len(spks),
        **r,
        "eer_threshold_sim": 1 - r["eer_threshold"],
        "at_configured_threshold": _rates(y, s, get_settings().voice_match_threshold),
    }
    _ = ds
    _save("speaker_librispeech", res)
    return res


def _save(name: str, res: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2))
    print(res.get("at_configured_threshold"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["face", "speaker"])
    ap.add_argument("--trials", type=int, default=2000)
    a = ap.parse_args(argv)
    face_lfw() if a.what == "face" else speaker_librispeech(a.trials)


if __name__ == "__main__":
    main()
