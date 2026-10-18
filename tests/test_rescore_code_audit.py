from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rescore_code_audit.py"
_spec = importlib.util.spec_from_file_location("rescore_code_audit", _SCRIPT)
rescore = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rescore)


def test_parse_steps_accepts_ranges_and_lists() -> None:
    assert rescore.parse_steps(None, [3, 1, 2]) == [1, 2, 3]
    assert rescore.parse_steps("1-3,5", [1, 2, 3, 4, 5]) == [1, 2, 3, 5]


def test_parse_steps_rejects_missing_and_malformed_steps() -> None:
    with pytest.raises(ValueError, match="without response records"):
        rescore.parse_steps("1-3", [1, 2])
    with pytest.raises(ValueError, match="Bad --steps"):
        rescore.parse_steps("one", [1])


def test_sample_id_round_trip_and_fingerprint() -> None:
    path = "../mopd/code/data/G-OPD-Training-Data/Eurus/code_train.parquet"
    fingerprint = hashlib.sha256(path.encode()).hexdigest()[:16]
    assert rescore.source_fingerprint(path) == fingerprint
    assert rescore.split_sample_id(f"code:{fingerprint}:1079") == ("code", fingerprint, 1079)


def test_discover_steps_needs_the_domain_records_file(tmp_path: Path) -> None:
    for step, domain in ((1, "code"), (2, "code"), (3, "math")):
        folder = tmp_path / f"step_{step:06d}" / "jsonls"
        folder.mkdir(parents=True)
        with gzip.open(folder / f"response_records_{domain}.jsonl.gz", "wt") as handle:
            handle.write(json.dumps({"sample_id": f"{domain}:abc:0"}) + "\n")

    assert sorted(rescore.discover_steps(tmp_path, "code")) == [1, 2]


def test_summarize_step_reports_overall_and_per_source_accuracy() -> None:
    rows = [
        {"score": 1.0, "data_source": "taco", "reward_timeout": 0.0, "reward_error": 0.0, "valid_token_count": 10},
        {"score": 0.0, "data_source": "taco", "reward_timeout": 1.0, "reward_error": 0.0, "valid_token_count": 30},
        {"score": 1.0, "data_source": "apps", "reward_timeout": 0.0, "reward_error": 0.0, "valid_token_count": 20},
        {"score": 0.0, "data_source": "apps", "reward_timeout": 0.0, "reward_error": 1.0, "valid_token_count": 20},
    ]

    summary = rescore.summarize_step(7, rows)

    assert summary["step"] == 7
    assert summary["n_samples"] == 4
    assert summary["n_pass"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["reward_timeout"] == 1
    assert summary["reward_error"] == 1
    assert summary["mean_response_tokens"] == 20.0
    assert summary["by_data_source"] == {
        "apps": {"n": 2, "accuracy": 0.5},
        "taco": {"n": 2, "accuracy": 0.5},
    }
