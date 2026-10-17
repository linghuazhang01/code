"""Budget mode "all": no top-k / top-p cap, every gated candidate is weighted."""

from pathlib import Path

import pytest
import yaml

from mopd_verl.domain_gradient.control_selection_budget import select_ranked_tokens
from mopd_verl.domain_gradient.control_selection_scoring import (
    ALL_BUDGET_MODE,
    ONLINE_CONTROL_BUDGET_MODES,
    normalize_online_budget_mode,
)
from mopd_verl.domain_gradient.control_selection_types import SelectedControlToken
from mopd_verl.domain_gradient.current_step_selection import select_current_step
from mopd_verl.domain_gradient.occurrence import Position
from mopd_verl.settings import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/token_selection/math_code/taxonomy"
BASE = CONFIGS / "mopd_math_code_toploss_m05_c01_math_control_code_progvocab_fixed4_8gpu_8c.yaml"


def _ranked(n: int) -> tuple[SelectedControlToken, ...]:
    return tuple(
        SelectedControlToken(
            token_id=300 + i, occurrence_count=1, mean_occurrences_per_step=1.0,
            mean_abs_loss=float(n - i), mean_selection_score=float(n - i),
            optimization_speed=None, observed_step_count=1,
        )
        for i in range(n)
    )


def test_all_is_a_registered_budget_mode() -> None:
    assert ALL_BUDGET_MODE == "all"
    assert ALL_BUDGET_MODE in ONLINE_CONTROL_BUDGET_MODES
    assert normalize_online_budget_mode(" ALL ") == ALL_BUDGET_MODE


def test_select_ranked_tokens_all_ignores_top_k_and_top_p() -> None:
    ranked = _ranked(7)
    assert select_ranked_tokens(ranked, budget_mode="top_k", top_k=2, top_p=0.05) == ranked[:2]
    assert (
        select_ranked_tokens(ranked, budget_mode="top_p", top_k=2, top_p=0.05, valid_token_count=100)
        == ranked[:5]
    )
    assert select_ranked_tokens(ranked, budget_mode="all", top_k=2, top_p=0.05) == ranked
    # No valid-token count is required under "all".
    assert select_ranked_tokens(ranked, budget_mode="all", top_k=1, top_p=0.01, valid_token_count=0) == ranked


def test_current_step_fraction_one_selects_every_gated_candidate() -> None:
    # This is what occurrence.py feeds select_current_step under budget mode "all".
    positions = [
        Position("math", 0, 0, c, t, loss)
        for c, (t, loss) in enumerate(zip([300, 300, 301, 302, 302, 999], [9.0, 1.0, 5.0, 0.5, 0.5, 7.0]))
    ]
    ranks = [({"math": 20}, positions, None)]
    result = select_current_step(
        ranks, {"math": [300, 301, 302]}, {"math": 1.0}, {"math": 0.0},
        0, True, unit="token_id", head_modes={"math": "top_loss"}, tail_mode="bottom_loss",
    )
    assert result.head == {"math": frozenset({(300,), (301,), (302,)})}
    assert result.metrics["math/current_step/head/budget_count"] == 20
    assert result.metrics["math/current_step/head/selected_count"] == 5
    # Occurrence gate still applies: 301 appears once and drops under minimum=2.
    gated = select_current_step(
        ranks, {"math": [300, 301, 302]}, {"math": 1.0}, {"math": 0.0},
        2, False, unit="token_id", head_modes={"math": "top_loss"}, tail_mode="bottom_loss",
    )
    assert gated.head == {"math": frozenset({(300,), (302,)})}


def test_unknown_budget_mode_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"extends": str(BASE), "audit": {"control_token_online_budget_mode": "everything"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="budget_mode must be one of"):
        load_config(path)
