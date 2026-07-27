from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mopd_verl.reproducibility import (
    GLOBAL_SEED_ENV,
    PYTHON_HASH_SEED_ENV,
    UINT32_MODULUS,
    derive_seed,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def test_derive_seed_is_rank_specific_and_numpy_compatible(self) -> None:
        self.assertEqual(derive_seed(42), 42)
        self.assertEqual(derive_seed(42, 3), 45)
        self.assertEqual(derive_seed(UINT32_MODULUS - 1, 2), 1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            derive_seed(-1)

    def test_seed_everything_covers_python_numpy_torch_and_cuda(self) -> None:
        numpy_seed = Mock()
        torch_seed = Mock()
        cuda_seed = Mock()
        fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=numpy_seed))
        fake_torch = SimpleNamespace(
            manual_seed=torch_seed,
            cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=cuda_seed),
        )

        with (
            patch.dict(sys.modules, {"numpy": fake_numpy, "torch": fake_torch}),
            patch.dict(os.environ, {}, clear=True),
            patch("random.seed") as python_seed,
        ):
            process_seed = seed_everything(42, rank=3)

            self.assertEqual(process_seed, 45)
            self.assertEqual(os.environ[GLOBAL_SEED_ENV], "42")
            self.assertEqual(os.environ[PYTHON_HASH_SEED_ENV], "45")
            python_seed.assert_called_once_with(45)
            numpy_seed.assert_called_once_with(45)
            torch_seed.assert_called_once_with(45)
            cuda_seed.assert_called_once_with(45)

    def test_distributed_seed_hooks_are_wired_into_training(self) -> None:
        launcher = (ROOT / "mopd_verl" / "launch.py").read_text(encoding="utf-8")
        main_ppo = (
            ROOT / "third_party" / "verl" / "verl" / "trainer" / "main_ppo.py"
        ).read_text(encoding="utf-8")
        worker = (
            ROOT
            / "third_party"
            / "verl"
            / "verl"
            / "single_controller"
            / "base"
            / "worker.py"
        ).read_text(encoding="utf-8")

        self.assertIn("actor_rollout_ref.rollout.seed={rollout.seed}", launcher)
        self.assertIn("+trainer.seed={trainer.seed}", launcher)
        self.assertIn("seed_everything(global_seed)", main_ppo)
        self.assertIn("seed_worker_from_environment(rank)", worker)


if __name__ == "__main__":
    unittest.main()
