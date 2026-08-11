from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import yaml

from mopd_verl.domain_gradient.config import DomainGradientConfig
from mopd_verl.settings import load_config


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "configs"
    / (
        "mopd_qwen4b_30b_a3b_instruct_2507_6gpu_math_code_science_"
        "topk32_reweight_control44.yaml"
    )
)


class FixedControlTokenWeightTests(unittest.TestCase):
    def _load_profile_with_weight(self, weight: float) -> float:
        payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
        payload["audit"]["control_token_loss_weight"] = weight
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "fixed-control-weight.yaml"
            config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )
            return load_config(config_path).audit.control_token_loss_weight

    def test_launcher_accepts_fractional_and_zero_weights(self) -> None:
        for weight in (0.5, 0.0):
            with self.subTest(weight=weight):
                self.assertEqual(self._load_profile_with_weight(weight), weight)

    def test_runtime_config_accepts_fractional_and_zero_weights(self) -> None:
        for weight in (0.5, 0.0):
            with self.subTest(weight=weight):
                config = DomainGradientConfig.from_meta(
                    {
                        "domains": ["math"],
                        "control_token_loss_weighting_enabled": True,
                        "control_token_loss_weight": weight,
                        "control_token_ids": [10],
                    }
                )
                self.assertEqual(config.control_token_weight, weight)

    def test_invalid_fixed_weights_are_rejected(self) -> None:
        for weight in (-0.1, math.inf, math.nan):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and non-negative",
                ):
                    self._load_profile_with_weight(weight)
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and non-negative",
                ):
                    DomainGradientConfig.from_meta(
                        {
                            "domains": ["math"],
                            "control_token_loss_weighting_enabled": True,
                            "control_token_loss_weight": weight,
                            "control_token_ids": [10],
                        }
                    )

    def test_zero_weight_masks_global_and_domain_control_tokens(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            self.skipTest(f"torch is unavailable: {exc}")

        token_ids = torch.tensor([[10, 20, 10], [10, 20, 30]])
        verl_module = ModuleType("verl")
        verl_module.__path__ = []
        utils_module = ModuleType("verl.utils")
        utils_module.__path__ = []
        device_module = ModuleType("verl.utils.device")
        device_module.get_device_id = lambda: torch.device("cpu")
        module_name = "mopd_verl.domain_gradient.token_weighting"
        saved_module = sys.modules.pop(module_name, None)
        try:
            with patch.dict(
                sys.modules,
                {
                    "verl": verl_module,
                    "verl.utils": utils_module,
                    "verl.utils.device": device_module,
                },
            ):
                from mopd_verl.domain_gradient.token_weighting import (
                    token_gradient_weights,
                )

                global_weights = token_gradient_weights(
                    token_ids,
                    control_token_ids=[10],
                    control_token_weight=0.0,
                    shared_token_ids=[],
                    shared_token_weight=1.0,
                )
        finally:
            sys.modules.pop(module_name, None)
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        torch.testing.assert_close(
            global_weights,
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        )

        from mopd_verl.domain_gradient.phase_control import phase_token_weights

        domain_weights = phase_token_weights(
            token_ids,
            torch.ones_like(token_ids),
            ["math", "code"],
            domain_token_ids={"math": [10], "code": [20]},
            control_weight=0.0,
            phase_enabled=False,
            span_enabled=False,
            phase_gates={},
            span_length=0,
            span_decay_tau=1.0,
            normalize_per_domain=False,
        )
        torch.testing.assert_close(
            domain_weights,
            torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        )

        normalized_domain_weights = phase_token_weights(
            token_ids,
            torch.ones_like(token_ids),
            ["math", "code"],
            domain_token_ids={"math": [10], "code": [20]},
            control_weight=0.0,
            phase_enabled=False,
            span_enabled=False,
            phase_gates={},
            span_length=0,
            span_decay_tau=1.0,
            normalize_per_domain=True,
        )
        torch.testing.assert_close(
            normalized_domain_weights,
            torch.tensor([[0.0, 3.0, 0.0], [1.5, 0.0, 1.5]]),
        )


if __name__ == "__main__":
    unittest.main()
