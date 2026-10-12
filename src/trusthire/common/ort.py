"""ONNX Runtime session factory: TensorRT → CUDA → DirectML → CPU, whichever is available."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import onnxruntime as ort

log = logging.getLogger(__name__)

_PREFERRED = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


def available_providers() -> list[str]:
    have = set(ort.get_available_providers())
    return [p for p in _PREFERRED if p in have]


def make_session(path: str | Path, fp16_trt: bool = True, threads: int | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is None:
        # Branches run concurrently; letting every session grab all cores oversubscribes the CPU badly
        # (measured: 9.5 s/window with default threads vs. far less with a small per-session pool).
        threads = int(
            os.environ.get("TRUSTHIRE_ORT_THREADS", str(max(1, min(3, (os.cpu_count() or 4) // 3))))
        )
    if threads:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    so.log_severity_level = 3
    # Denormal activations make these CNNs ~40x slower on x86 CPUs (measured: EffNet-B4 11.8 s -> 0.27 s).
    so.add_session_config_entry("session.set_denormal_as_zero", "1")
    providers: list[str | tuple[str, dict[str, object]]] = []
    for p in available_providers():
        if p == "TensorrtExecutionProvider":
            providers.append(
                (
                    p,
                    {
                        "trt_fp16_enable": fp16_trt,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(Path(path).parent / "trt_cache"),
                    },
                )
            )
        else:
            providers.append(p)
    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    log.debug("ORT %s -> %s", Path(path).name, sess.get_providers())
    return sess
