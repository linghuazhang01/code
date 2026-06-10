import importlib.util
import os
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def load_constants_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "verl"
        / "verl"
        / "trainer"
        / "constants_ppo.py"
    )
    spec = importlib.util.spec_from_file_location("vendored_verl_constants_ppo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerlRayRuntimeEnvTests(unittest.TestCase):
    def test_runtime_env_omits_null_working_dir(self) -> None:
        module = load_constants_module()

        with patch.dict(os.environ, {}, clear=True):
            runtime_env = module.get_ppo_ray_runtime_env()

        self.assertNotIn("working_dir", runtime_env)
        self.assertEqual(runtime_env["env_vars"]["NCCL_DEBUG"], "WARN")
        self.assertNotIn('"working_dir": None', Path(module.__file__).read_text(encoding="utf-8"))

    def test_existing_environment_values_are_not_repeated(self) -> None:
        module = load_constants_module()

        with patch.dict(os.environ, {"NCCL_DEBUG": "INFO"}, clear=True):
            runtime_env = module.get_ppo_ray_runtime_env()

        self.assertNotIn("NCCL_DEBUG", runtime_env["env_vars"])


if __name__ == "__main__":
    unittest.main()
