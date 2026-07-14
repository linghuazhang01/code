from __future__ import annotations

from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from typing import Callable, cast
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_verifier() -> Callable[[Path], tuple[Path, ...]]:
    script = ROOT / "scripts" / "apply_gopd_audit_patch.py"
    return cast(
        Callable[[Path], tuple[Path, ...]],
        run_path(str(script))["verify_integration"],
    )


class IntegrationVerifierTests(unittest.TestCase):
    def test_reviewed_integration_verifier_is_read_only(self) -> None:
        verify = _load_verifier()
        integration_root = ROOT / "third_party"
        before = {path: path.read_bytes() for path in verify(integration_root)}

        verified_again = verify(integration_root)

        self.assertEqual(
            {path: path.read_bytes() for path in verified_again},
            before,
        )

    def test_pristine_actor_fails_before_any_mutation(self) -> None:
        verify = _load_verifier()
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            actor = root / "verl/verl/workers/actor/dp_actor.py"
            actor.parent.mkdir(parents=True)
            actor.write_text(
                "class DataParallelPPOActor: pass\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "(?i)reviewed"):
                verify(root)

            self.assertEqual(
                actor.read_text(encoding="utf-8"),
                "class DataParallelPPOActor: pass\n",
            )


if __name__ == "__main__":
    unittest.main()
