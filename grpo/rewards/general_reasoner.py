"""General-Reasoner rule reward adapter for validation/fallback scoring."""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    try:
        from math_verify.errors import TimeoutException
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except ImportError as exc:
        raise RuntimeError("Install math-verify to use the General-Reasoner rule reward adapter.") from exc

    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        score, _ = verify_func([ground_truth_boxed], [solution_str])
    except TimeoutException:
        return 0.0
    except Exception:
        return 0.0
    return float(score)
