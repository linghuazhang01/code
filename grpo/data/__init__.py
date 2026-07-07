"""Data preparation helpers for M2RL-style GRPO recipes."""

from grpo.data.m2rl import (
    M2RLSchemaReport,
    m2rl_frame_to_verl,
    m2rl_to_verl_parquet,
    validate_m2rl_frame,
    validate_m2rl_parquet,
)
from grpo.data.nemotron import SplitOutputs, normalize_nemotron_record, prepare_nemotron_rl_data

__all__ = [
    "M2RLSchemaReport",
    "SplitOutputs",
    "m2rl_frame_to_verl",
    "m2rl_to_verl_parquet",
    "normalize_nemotron_record",
    "prepare_nemotron_rl_data",
    "validate_m2rl_frame",
    "validate_m2rl_parquet",
]
