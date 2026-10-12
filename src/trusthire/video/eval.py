"""Video deepfake evaluation: backbone comparison, cross-dataset generalisation and robustness.

    python -m trusthire.video.eval --crops crops.csv --backbones effnb4 xception --test-datasets ffpp celebdf dfdc \
        --conditions clean jpeg_q30 lowres_x3 dark_g2.2

Video-level score = mean of frame P(fake) over the video's crops (the live detector's top-half mean is also
reported). Writes results/video/benchmark.{json,md}. Standard protocol: train on FF++ (c23), test on
FF++ test and — cross-dataset — on Celeb-DF v2 test / DFDC.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd

from trusthire.common.augment import VIDEO_CONDITIONS
from trusthire.common.settings import REPO_ROOT
from trusthire.evaluation.metrics import report
from trusthire.video.infer import VideoDeepfakeDetector

OUT = REPO_ROOT / "results" / "video"


def evaluate(
    crops_csv: str,
    backbones: list[str],
    datasets: list[str],
    conditions: list[str],
    max_videos: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(crops_csv)
    df = df[df.split == "test"] if "split" in df else df
    res: dict[str, Any] = {}
    for bb in backbones:
        det = VideoDeepfakeDetector(bb)
        for ds in datasets:
            d = df[df.dataset == ds]
            vids = d.video_id.unique()
            if max_videos and len(vids) > max_videos:
                rng = np.random.default_rng(seed)
                # stratified subsample
                lab = d.groupby("video_id").label.first()
                pick = [
                    *rng.choice(lab[lab == 0].index, min(max_videos // 2, (lab == 0).sum()), replace=False),
                    *rng.choice(lab[lab == 1].index, min(max_videos // 2, (lab == 1).sum()), replace=False),
                ]
                d = d[d.video_id.isin(pick)]
            for cond in conditions:
                ys, mean_s, top_s, times = [], [], [], []
                for _vid, g in d.groupby("video_id"):
                    imgs = [VIDEO_CONDITIONS[cond](cv2.imread(p)) for p in g.crop]
                    t0 = time.perf_counter()
                    probs, _ = det.frame_scores(imgs)
                    times.append((time.perf_counter() - t0) / len(imgs))
                    ys.append(int(g.label.iloc[0]))
                    mean_s.append(float(probs.mean()))
                    k = max(1, len(probs) // 2)
                    top_s.append(float(np.sort(probs)[-k:].mean()))
                if len(set(ys)) < 2:
                    continue
                r = report(np.array(ys), np.array(mean_s), bootstrap=500).as_dict()
                r_top = report(np.array(ys), np.array(top_s))
                r.update(
                    {
                        "backbone": bb,
                        "dataset": ds,
                        "condition": cond,
                        "n_videos": len(ys),
                        "auc_topk": r_top.auc,
                        "ms_per_frame": 1e3 * float(np.median(times)),
                    }
                )
                res[f"{bb}/{ds}/{cond}"] = r
                print(
                    f"{bb:10s} {ds:8s} {cond:12s} AUC={r['auc']:.4f} EER={100 * r['eer']:.1f}% n={len(ys)}",
                    flush=True,
                )
    (OUT / "benchmark.json").write_text(json.dumps(res, indent=2))
    lines = [
        "| Backbone | Test set | Condition | Videos | AUC [95% CI] | EER % | AUC (top-k) | ms/frame |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(res.values(), key=lambda r: (r["dataset"], r["condition"], -r["auc"])):
        ci = r["auc_ci"]
        lines.append(
            f"| {r['backbone']} | {r['dataset']} | {r['condition']} | {r['n_videos']} | {r['auc']:.4f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {100 * r['eer']:.1f} | {r['auc_topk']:.4f} | {r['ms_per_frame']:.0f} |"
        )
    (OUT / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--backbones", nargs="+", default=["effnb4", "xception"])
    ap.add_argument("--test-datasets", nargs="+", default=["ffpp", "celebdf"])
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--max-videos", type=int)
    a = ap.parse_args(argv)
    evaluate(a.crops, a.backbones, a.test_datasets, a.conditions, a.max_videos)


if __name__ == "__main__":
    main()
