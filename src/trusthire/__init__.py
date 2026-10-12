"""TrustHire — multimodal fake-interview detection."""

__version__ = "0.1.0"


def _flush_denormals() -> None:
    # Same reason as in common/ort.py: denormal floats cripple CPU conv throughput.
    try:
        import torch

        torch.set_flush_denormal(True)
        if not torch.cuda.is_available():
            import os

            torch.set_num_threads(int(os.environ.get("TRUSTHIRE_TORCH_THREADS", "3")))
    except Exception:  # torch not importable (e.g. stdlib-only registry use)
        pass


_flush_denormals()
