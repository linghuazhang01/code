from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = ROOT / "third_party" / "verl"


class EntropyNumericsTests(unittest.TestCase):
    def test_chunked_bf16_entropy_uses_stable_fp32_formula(self) -> None:
        try:
            import torch

            sys.path.insert(0, str(VERL_ROOT))
            from verl.utils.torch_functional import (
                entropy_from_logits_with_chunking,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"verl test dependencies are unavailable: {exc}")

        logits = torch.tensor(
            [
                [-128.0, -111.5],
                [48.0, 12.0],
                [-20.0, -26.125],
            ],
            dtype=torch.bfloat16,
        )
        entropy = entropy_from_logits_with_chunking(logits, chunk_size=2)
        reference_log_probs = torch.log_softmax(logits.float(), dim=-1)
        reference = -(reference_log_probs.exp() * reference_log_probs).sum(dim=-1)

        self.assertEqual(entropy.dtype, torch.float32)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue((entropy >= 0.0).all())
        torch.testing.assert_close(entropy, reference, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
