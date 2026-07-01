from __future__ import annotations

import random
import time

import numpy as np


def normalize_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return int(time.time() * 1000) % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
