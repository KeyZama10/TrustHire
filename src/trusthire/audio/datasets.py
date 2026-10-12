"""Anti-spoofing datasets.

Public benchmarks are pulled as Parquet shards from the SpeechAntiSpoofingBenchmarks HF mirrors
(In-the-Wild: Apache-2.0; ASVspoof 2019 LA / 2021 LA: ODC-By). Downloading a subset of shards gives a
*subset* evaluation — reports always state how many shards/utterances were used.

For training (ASVspoof 2019 LA train split) use the official Edinburgh DataShare release:
  https://datashare.ed.ac.uk/handle/10283/3336  -> scripts/data/get_asvspoof2019.sh
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from trusthire.common.media import to_mono_16k
from trusthire.common.registry import data_dir

BENCH_REPOS = {
    "itw": "SpeechAntiSpoofingBenchmarks/InTheWild",
    "asv19la": "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA",
    "asv21la": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA",
    "asv21df": "SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
}

DATA_ROOT = data_dir() / "public" / "audio"


@dataclass
class Utterance:
    uid: str
    audio: np.ndarray  # float32 mono 16 kHz
    label: int  # 1 = spoof / fake, 0 = bona fide  (TrustHire convention: 1 = suspicious)
    meta: dict[str, str]


def fetch_shards(name: str, n_shards: int = 1, shard_ids: list[int] | None = None) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    repo = BENCH_REPOS[name]
    files = sorted(
        f
        for f in HfApi().list_repo_files(repo, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet") and "labels" not in f
    )
    if shard_ids is None:
        # spread picks across the split so we don't only get one speaker block
        step = max(1, len(files) // n_shards)
        shard_ids = list(range(0, len(files), step))[:n_shards]
    out = []
    for i in shard_ids:
        out.append(Path(hf_hub_download(repo, files[i], repo_type="dataset", local_dir=DATA_ROOT / name)))
    return out


def _label_of(row: pd.Series) -> int:
    for col in ("label", "key", "target", "is_spoof"):
        if col in row:
            v = row[col]
            if isinstance(v, str):
                return 0 if v.lower() in ("bonafide", "bona-fide", "bona fide", "real", "genuine") else 1
            return int(v)
    raise KeyError(f"no label column in {list(row.index)}")


def iter_parquet(paths: list[Path], max_per_shard: int | None = None, seed: int = 0) -> Iterator[Utterance]:
    rng = np.random.default_rng(seed)
    for p in paths:
        df = pd.read_parquet(p)
        if max_per_shard and len(df) > max_per_shard:
            df = df.iloc[np.sort(rng.choice(len(df), max_per_shard, replace=False))]
        audio_col = next(c for c in df.columns if c in ("audio", "wav", "flac"))
        for _, row in df.iterrows():
            a = row[audio_col]
            raw = a["bytes"] if isinstance(a, dict) else a
            x, sr = sf.read(io.BytesIO(raw), dtype="float32")
            meta = {
                k: str(row[k])
                for k in df.columns
                if k not in (audio_col,) and not isinstance(row[k], (bytes, dict))
            }
            uid = str(row.get("utt_id", row.get("file", row.get("id", len(meta)))))
            yield Utterance(uid, to_mono_16k(x, sr), _label_of(row), meta)


def bona_fide_label_convention_note() -> str:
    return "label 1 = spoof (suspicious), 0 = bona fide"
