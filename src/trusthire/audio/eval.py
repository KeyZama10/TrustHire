"""Benchmark voice anti-spoofing backends on public data subsets (+ degradation conditions).

    python -m trusthire.audio.eval --datasets itw asv19la --n 400 --backends aasist aasist_l spectra0 \
        --conditions clean opus_16k noise_snr10

Each utterance is scored on its first 64600 samples (4.04 s, tiled if shorter) — the same context the live
pipeline uses. Results: results/audio/benchmark.{json,md} and per-utterance score CSVs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trusthire.audio.datasets import DATA_ROOT, Utterance, fetch_shards, iter_parquet
from trusthire.audio.infer import build_backend, tile
from trusthire.common.augment import AUDIO_CONDITIONS
from trusthire.common.settings import REPO_ROOT
from trusthire.evaluation.metrics import report

OUT = REPO_ROOT / "results" / "audio"


def stratified_subset(name: str, n: int, seed: int = 0) -> list[Utterance]:
    shards = sorted((DATA_ROOT / name / "data").glob("test-*.parquet")) or fetch_shards(name, 2)
    utts = list(iter_parquet(shards, max_per_shard=max(n * 2, 400), seed=seed))
    rng = np.random.default_rng(seed)
    pos = [u for u in utts if u.label == 1]
    neg = [u for u in utts if u.label == 0]
    k = min(n // 2, len(pos), len(neg))
    pick = [pos[i] for i in rng.choice(len(pos), k, replace=False)] + [
        neg[i] for i in rng.choice(len(neg), k, replace=False)
    ]
    return pick


def run(
    datasets: list[str], backends: list[str], conditions: list[str], n: int, seed: int = 0
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        utts = stratified_subset(ds, n, seed)
        shards = sorted(p.name for p in (DATA_ROOT / ds / "data").glob("test-*.parquet"))
        for cond in conditions:
            rng = np.random.default_rng(seed)
            xs = np.stack([tile(AUDIO_CONDITIONS[cond](u.audio, rng)) for u in utts])
            y = np.array([u.label for u in utts])
            for be_name in backends:
                key = f"{ds}/{cond}/{be_name}"
                csv = OUT / f"scores_{ds}_{cond}_{be_name.replace('+', '-')}.csv"
                if csv.exists():  # cached scores from a previous run
                    s = pd.read_csv(csv)["score"].to_numpy()
                    lj = OUT / "latency.json"
                    lat = float(json.loads(lj.read_text()).get(be_name, np.nan)) if lj.exists() else np.nan
                else:
                    be = build_backend(be_name)
                    be.p_spoof(xs[:1])  # warm-up
                    s, times = [], []
                    for i in range(0, len(xs), 8):
                        t0 = time.perf_counter()
                        s.extend(be.p_spoof(xs[i : i + 8]).tolist())
                        times.append((time.perf_counter() - t0) / len(xs[i : i + 8]))
                    s = np.array(s)
                    lat = float(np.median(times) * 1e3)
                    pd.DataFrame({"uid": [u.uid for u in utts], "label": y, "score": s}).to_csv(
                        csv, index=False
                    )
                    lj = OUT / "latency.json"
                    if (
                        cond == "clean"
                    ):  # degradations (codecs, reverb) add their own cost; time clean audio only
                        d = json.loads(lj.read_text()) if lj.exists() else {}
                        d[be_name] = lat
                        lj.write_text(json.dumps(d, indent=2))
                    else:
                        lat = (
                            float(json.loads(lj.read_text()).get(be_name, np.nan)) if lj.exists() else np.nan
                        )
                r = report(y, s, bootstrap=500, seed=seed).as_dict()
                r.update(
                    {
                        "dataset": ds,
                        "condition": cond,
                        "backend": be_name,
                        "latency_ms_per_utt": lat,
                        "shards": shards,
                        "subset": f"stratified {len(utts)} utts (seed {seed})",
                    }
                )
                results[key] = r
                eer_pct = 100 * float(r["eer"])  # type: ignore[arg-type]
                print(
                    f"{key:40s} AUC={r['auc']:.4f} EER={eer_pct:.2f}% n={r['n']} lat={lat:.0f}ms",
                    flush=True,
                )
    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, OUT / "benchmark.md")
    return results


def write_markdown(results: dict, path: Path) -> None:
    rows = sorted(results.values(), key=lambda r: (r["dataset"], r["condition"], r["eer"]))
    lines = [
        "| Dataset | Condition | Backend | n | AUC [95% CI] | EER % [95% CI] | TPR@1%FPR | ECE | CPU ms/utt |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ac, ec = r.get("auc_ci") or (np.nan, np.nan), r.get("eer_ci") or (np.nan, np.nan)
        lines.append(
            f"| {r['dataset']} | {r['condition']} | {r['backend']} | {r['n']} | {r['auc']:.4f} "
            f"[{ac[0]:.3f}, {ac[1]:.3f}] | {100 * r['eer']:.2f} [{100 * ec[0]:.1f}, {100 * ec[1]:.1f}] | "
            f"{r['tpr_at_fpr1']:.3f} | {r['ece']:.3f} | {r['latency_ms_per_utt']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rebuild_from_cache(seed: int = 0) -> dict[str, Any]:
    """Recompute benchmark.{json,md} from every cached per-utterance score CSV (no model inference)."""
    from trusthire.audio.datasets import BENCH_REPOS

    lj = OUT / "latency.json"
    lat = json.loads(lj.read_text()) if lj.exists() else {}
    lat.pop("_conditions", None)
    results: dict[str, dict[str, Any]] = {}
    conds = sorted(AUDIO_CONDITIONS, key=len, reverse=True)  # match longest condition names first
    for csv in sorted(OUT.glob("scores_*.csv")):
        rest = csv.stem.removeprefix("scores_")
        ds = next((d for d in BENCH_REPOS if rest.startswith(d + "_")), None)
        if ds is None:
            continue
        rest = rest[len(ds) + 1 :]
        cond = next((c for c in conds if rest.startswith(c + "_")), None)
        if cond is None:
            continue
        backend = rest[len(cond) + 1 :].replace("-", "+")
        df = pd.read_csv(csv)
        r = report(df["label"].to_numpy(), df["score"].to_numpy(), bootstrap=500, seed=seed).as_dict()
        r.update(
            {
                "dataset": ds,
                "condition": cond,
                "backend": backend,
                "latency_ms_per_utt": lat.get(backend, np.nan),
                "subset": f"stratified {len(df)} utts (seed {seed})",
            }
        )
        results[f"{ds}/{cond}/{backend}"] = r
    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, OUT / "benchmark.md")
    print(f"rebuilt {len(results)} rows -> {OUT / 'benchmark.md'}")
    return results


def measure_latency(backends: list[str], n: int = 40) -> dict[str, float]:
    """Dedicated CPU latency measurement (run on an otherwise idle machine)."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, 64600)) * 0.05).astype(np.float32)
    out: dict[str, float] = {}
    for b in backends:
        be = build_backend(b)
        be.p_spoof(x[:1])
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            be.p_spoof(x[i : i + 1])
            times.append(time.perf_counter() - t0)
        out[b] = 1e3 * float(np.median(times))
        print(f"{b}: {out[b]:.0f} ms/utterance (median of {n})")
    import psutil

    bat = psutil.sensors_battery()
    freq = psutil.cpu_freq()
    meta = {
        "on_battery": bool(bat and not bat.power_plugged),
        "cpu_mhz": freq.current if freq else None,
        "measured": time.strftime("%Y-%m-%d %H:%M"),
    }
    (OUT / "latency.json").write_text(json.dumps({**out, "_conditions": meta}, indent=2))
    if meta["on_battery"]:
        print("WARNING: measured on battery power — CPU is throttled; re-measure plugged in")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-latency", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="recompute tables from cached score CSVs")
    ap.add_argument("--datasets", nargs="+", default=["itw", "asv19la"])
    ap.add_argument("--backends", nargs="+", default=["aasist", "aasist_l", "spectra0"])
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.measure_latency:
        measure_latency(a.backends)
        return
    if a.rebuild:
        rebuild_from_cache(a.seed)
        return
    run(a.datasets, a.backends, a.conditions, a.n, a.seed)


if __name__ == "__main__":
    main()
