"""Generic ONNX export with numerical parity check against PyTorch."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def export_onnx(
    model: torch.nn.Module,
    example: tuple[torch.Tensor, ...],
    out_path: str | Path,
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_batch: bool = True,
    opset: int = 18,
    atol: float = 1e-3,
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    dyn = {n: {0: "batch"} for n in [*input_names, *output_names]} if dynamic_batch else None
    tmp = out.with_suffix(".tmp.onnx")
    try:
        # legacy TorchScript exporter: most robust for these CNN/graph models
        torch.onnx.export(
            model,
            example,
            str(tmp),
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dyn,
            opset_version=opset,
            dynamo=False,
        )
    except Exception as e:  # pragma: no cover - depends on torch version
        log.warning("legacy exporter failed (%s); trying dynamo exporter", e)
        batch = torch.export.Dim("batch", min=1, max=64)
        shapes = tuple({0: batch} for _ in example) if dynamic_batch else None
        prog = torch.onnx.export(
            model,
            example,
            dynamo=True,
            dynamic_shapes=shapes,
            input_names=list(input_names),
            output_names=list(output_names),
        )
        assert prog is not None
        prog.save(str(tmp))
    _check_parity(model, example, tmp, input_names, atol)
    tmp.replace(out)
    for extra in tmp.parent.glob(tmp.name + ".data"):
        extra.replace(out.with_suffix(".onnx.data"))
    return out


def _check_parity(
    model: torch.nn.Module,
    example: tuple[torch.Tensor, ...],
    path: Path,
    input_names: Sequence[str],
    atol: float,
) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        ref = model(*example)
    ref_t = ref if isinstance(ref, (tuple, list)) else (ref,)
    got = sess.run(None, {n: t.numpy() for n, t in zip(input_names, example, strict=True)})
    for r, g in zip(ref_t, got, strict=False):
        err = float(np.max(np.abs(r.numpy() - g)))
        if err > atol * max(1.0, float(np.max(np.abs(r.numpy())))):
            raise RuntimeError(f"ONNX parity failed for {path.name}: max abs err {err:.2e}")
    log.info("ONNX parity ok: %s", path.name)
