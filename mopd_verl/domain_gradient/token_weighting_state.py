"""Serializable state and configuration constants for token weighting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PER_STEP_MEAN_ABS_LOSS_SELECTION = "per_step_mean_abs_loss"
CUMULATIVE_ABS_LOSS_SELECTION = "cumulative_abs_loss"
SHARED_TOKEN_SELECTION_MODES = frozenset(
    {
        PER_STEP_MEAN_ABS_LOSS_SELECTION,
        CUMULATIVE_ABS_LOSS_SELECTION,
    }
)


@dataclass(frozen=True)
class CumulativeTokenLossState:
    """Serializable cumulative absolute-loss statistics by domain and token."""

    domains: tuple[str, ...]
    statistics: tuple[
        tuple[str, tuple[tuple[int, float, int], ...]],
        ...,
    ]
    last_updated_step: int | None = None
    selection_mode: str = CUMULATIVE_ABS_LOSS_SELECTION

    def statistics_map(self) -> dict[str, dict[int, tuple[float, int]]]:
        return {
            domain: {
                int(token_id): (float(loss_abs_sum), int(count))
                for token_id, loss_abs_sum, count in rows
            }
            for domain, rows in self.statistics
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "domains": self.domains,
            "statistics": self.statistics,
            "last_updated_step": self.last_updated_step,
            "selection_mode": self.selection_mode,
        }

    def domain_summaries(self) -> dict[str, dict[str, float]]:
        return {
            domain: {
                "observed_token_type_count": float(len(rows)),
                "cumulative_abs_loss_mass": float(
                    sum(loss_abs_sum for _, loss_abs_sum, _ in rows)
                ),
                "cumulative_occurrence_count": float(
                    sum(count for _, _, count in rows)
                ),
            }
            for domain, rows in self.statistics
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CumulativeTokenLossState":
        domains = tuple(str(domain) for domain in value.get("domains", ()))
        raw_statistics = value.get("statistics", ())
        statistics = tuple(
            (
                str(domain),
                tuple(
                    (
                        int(token_id),
                        float(loss_abs_sum),
                        int(count),
                    )
                    for token_id, loss_abs_sum, count in rows
                ),
            )
            for domain, rows in raw_statistics
        )
        if tuple(domain for domain, _ in statistics) != domains:
            raise ValueError(
                "Serialized cumulative token-loss domains do not match."
            )
        for _, rows in statistics:
            if any(
                token_id < 0
                or loss_abs_sum < 0.0
                or not math.isfinite(loss_abs_sum)
                or count < 1
                for token_id, loss_abs_sum, count in rows
            ):
                raise ValueError(
                    "Serialized cumulative token-loss statistics are invalid."
                )
        raw_step = value.get("last_updated_step")
        selection_mode = str(
            value.get(
                "selection_mode",
                CUMULATIVE_ABS_LOSS_SELECTION,
            )
        )
        if selection_mode != CUMULATIVE_ABS_LOSS_SELECTION:
            raise ValueError(
                "Serialized cumulative token-loss selection mode is invalid."
            )
        return cls(
            domains=domains,
            statistics=statistics,
            last_updated_step=None if raw_step is None else int(raw_step),
            selection_mode=selection_mode,
        )


def initial_cumulative_token_loss_state(
    domains: Sequence[str],
) -> CumulativeTokenLossState:
    normalized = tuple(dict.fromkeys(str(domain) for domain in domains))
    return CumulativeTokenLossState(
        domains=normalized,
        statistics=tuple((domain, tuple()) for domain in normalized),
    )
