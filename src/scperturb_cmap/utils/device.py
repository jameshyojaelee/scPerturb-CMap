from __future__ import annotations

from typing import Literal

DeviceLiteral = Literal["cuda", "mps", "cpu"]


def get_device() -> DeviceLiteral:
    """Return the best available compute device.

    Preference order: "cuda" (NVIDIA GPU) > "mps" (Apple Silicon) > "cpu".
    The function fails safe to "cpu" if PyTorch is unavailable.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    try:
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            return "cuda"
    except Exception:
        # If CUDA checks raise, fall back to other options
        pass

    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and hasattr(mps, "is_available") and mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"
