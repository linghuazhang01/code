from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import torch.distributed.fsdp  # noqa: F401


class FSDPCheckpointTopologyTests(unittest.TestCase):
    @contextmanager
    def _load_module(self) -> Iterator[Any]:
        root = Path(__file__).resolve().parents[1]
        module_path = (
            root
            / "third_party"
            / "verl"
            / "verl"
            / "utils"
            / "checkpoint"
            / "fsdp_checkpoint_topology.py"
        )
        verl_module = ModuleType("verl")
        verl_module.__path__ = []
        utils_module = ModuleType("verl.utils")
        utils_module.__path__ = []
        device_module = ModuleType("verl.utils.device")
        device_module.is_cuda_available = False
        fs_module = ModuleType("verl.utils.fs")
        fs_module.exists = lambda path: Path(path).exists()
        fs_module.copy_to_local = lambda path: path
        fsdp_utils_module = ModuleType("verl.utils.fsdp_utils")
        fsdp_utils_module.fsdp_version = lambda model: 1
        module_name = "_opd_fsdp_checkpoint_topology_test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {module_path}")
        module = importlib.util.module_from_spec(spec)
        stubs = {
            "verl": verl_module,
            "verl.utils": utils_module,
            "verl.utils.device": device_module,
            "verl.utils.fs": fs_module,
            "verl.utils.fsdp_utils": fsdp_utils_module,
            module_name: module,
        }
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
            yield module
        sys.modules.pop(module_name, None)

    @staticmethod
    def _write_metadata(path: str, **overrides: Any) -> None:
        metadata = {
            "FSDP_version": 1,
            "world_size": 2,
            "schema_version": 2,
            "effective_sharding_strategy": "NO_SHARD",
            "fsdp_process_group_size": 2,
        }
        metadata.update(overrides)
        with open(
            Path(path) / "fsdp_config.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle)

    def test_matching_topology_is_accepted(self) -> None:
        with self._load_module() as module, tempfile.TemporaryDirectory() as path:
            self._write_metadata(path)
            runtime = module.FSDPConfig(
                FSDP_version=1,
                world_size=2,
                effective_sharding_strategy="NO_SHARD",
                fsdp_process_group_size=2,
            )

            module.validate_checkpoint_topology(path, runtime)

    def test_size_one_to_size_two_restore_is_rejected(self) -> None:
        with self._load_module() as module, tempfile.TemporaryDirectory() as path:
            self._write_metadata(path)
            runtime = module.FSDPConfig(
                FSDP_version=1,
                world_size=2,
                effective_sharding_strategy="FULL_SHARD",
                fsdp_process_group_size=2,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "Cross-topology FSDP checkpoint restore is unsupported",
            ):
                module.validate_checkpoint_topology(path, runtime)

    def test_remote_metadata_is_copied_before_validation(self) -> None:
        with self._load_module() as module, tempfile.TemporaryDirectory() as path:
            self._write_metadata(path)
            local_config = str(Path(path) / "fsdp_config.json")
            runtime = module.FSDPConfig(
                FSDP_version=1,
                world_size=2,
                effective_sharding_strategy="NO_SHARD",
                fsdp_process_group_size=2,
            )
            with (
                patch.object(module, "exists", return_value=True) as exists_mock,
                patch.object(
                    module,
                    "copy_to_local",
                    return_value=local_config,
                ) as copy_mock,
            ):
                module.validate_checkpoint_topology("hdfs://run/step_4", runtime)

            remote_config = "hdfs://run/step_4/fsdp_config.json"
            exists_mock.assert_called_once_with(remote_config)
            copy_mock.assert_called_once_with(remote_config)

    def test_legacy_and_missing_metadata_warn_without_breaking_restore(self) -> None:
        with self._load_module() as module, tempfile.TemporaryDirectory() as path:
            runtime = module.FSDPConfig(FSDP_version=1, world_size=2)
            with self.assertWarnsRegex(UserWarning, "no fsdp_config.json"):
                module.validate_checkpoint_topology(path, runtime)

            self._write_metadata(path)
            config_path = Path(path) / "fsdp_config.json"
            with open(config_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            metadata.pop("effective_sharding_strategy")
            metadata.pop("fsdp_process_group_size")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle)
            with self.assertWarnsRegex(UserWarning, "Legacy FSDP checkpoint"):
                module.validate_checkpoint_topology(path, runtime)

    def test_state_dict_mode_matches_effective_fsdp1_strategy(self) -> None:
        with self._load_module() as module:
            class FakeFSDP:
                def __init__(self, strategy: Any) -> None:
                    self.sharding_strategy = strategy

            module.FSDP = FakeFSDP
            no_shard_model = FakeFSDP(module.ShardingStrategy.NO_SHARD)
            full_shard_model = FakeFSDP(module.ShardingStrategy.FULL_SHARD)

            full_type, full_model_cfg, full_optim_cfg = (
                module.checkpoint_state_dict_settings(
                    no_shard_model,
                    include_model=True,
                    include_optimizer=True,
                )
            )
            self.assertEqual(full_type, module.StateDictType.FULL_STATE_DICT)
            self.assertTrue(full_model_cfg.offload_to_cpu)
            self.assertFalse(full_model_cfg.rank0_only)
            self.assertTrue(full_optim_cfg.offload_to_cpu)
            self.assertFalse(full_optim_cfg.rank0_only)

            sharded_type, sharded_model_cfg, sharded_optim_cfg = (
                module.checkpoint_state_dict_settings(
                    full_shard_model,
                    include_model=True,
                    include_optimizer=True,
                )
            )
            self.assertEqual(
                sharded_type,
                module.StateDictType.SHARDED_STATE_DICT,
            )
            self.assertFalse(sharded_model_cfg.offload_to_cpu)
            self.assertFalse(sharded_optim_cfg.offload_to_cpu)


if __name__ == "__main__":
    unittest.main()
