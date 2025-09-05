from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility.

    If PyTorch is unavailable, silently skips the torch-specific parts.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        # Torch not installed or other issue; skip
        pass

