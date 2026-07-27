"""Seed helpers shared by the launcher, Ray controller, and distributed workers."""

from __future__ import annotations

import os
import random

GLOBAL_SEED_ENV = "MOPD_GLOBAL_SEED"
PYTHON_HASH_SEED_ENV = "PYTHONHASHSEED"
UINT32_MODULUS = 2**32


def derive_seed(seed: int, offset: int = 0) -> int:
    """Derive a NumPy-compatible process seed from a base seed and offset."""
    base_seed = int(seed)
    if base_seed < 0:
        raise ValueError("seed must be non-negative")
    return (base_seed + int(offset)) % UINT32_MODULUS


def seed_everything(seed: int, *, rank: int = 0) -> int:
    """Seed Python, NumPy, Torch, and every visible CUDA device."""
    process_seed = derive_seed(seed, rank)
    os.environ[GLOBAL_SEED_ENV] = str(int(seed))
    os.environ[PYTHON_HASH_SEED_ENV] = str(process_seed)

    random.seed(process_seed)

    import numpy as np
    import torch

    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)
    return process_seed


def seed_worker_from_environment(rank: int) -> int | None:
    """Seed a distributed worker when the launcher exported a global seed."""
    raw_seed = os.getenv(GLOBAL_SEED_ENV)
    if raw_seed is None:
        return None
    return seed_everything(int(raw_seed), rank=rank)
