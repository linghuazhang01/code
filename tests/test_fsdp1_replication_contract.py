from __future__ import annotations

import unittest
from pathlib import Path


class FSDP1ReplicationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_size_one_uses_world_no_shard(self) -> None:
        source = (
            self.root
            / "third_party"
            / "verl"
            / "verl"
            / "utils"
            / "fsdp_mesh.py"
        ).read_text(encoding="utf-8")

        self.assertIn("if fsdp_size == 1:", source)
        self.assertIn("return None, ShardingStrategy.NO_SHARD", source)
        self.assertIn("handle.uses_sharded_strategy", source)
        self.assertIn("validate_fsdp1_size_one_topology", source)
        self.assertIn("Expected NO_SHARD on the worker WORLD process group", source)

    def test_all_fsdp1_builders_use_the_shared_resolution(self) -> None:
        worker_source = (
            self.root
            / "third_party"
            / "verl"
            / "verl"
            / "workers"
            / "fsdp_workers.py"
        ).read_text(encoding="utf-8")
        engine_source = (
            self.root
            / "third_party"
            / "verl"
            / "verl"
            / "workers"
            / "engine"
            / "fsdp"
            / "transformer_impl.py"
        ).read_text(encoding="utf-8")

        self.assertGreaterEqual(
            worker_source.count("resolve_fsdp1_mesh_and_strategy("),
            3,
        )
        self.assertIn("resolve_fsdp1_mesh_and_strategy(", engine_source)
        self.assertIn("validate_fsdp1_size_one_topology(", worker_source)
        self.assertIn("validate_fsdp1_size_one_topology(", engine_source)
        self.assertGreaterEqual(worker_source.count("mesh=init_mesh"), 3)
        self.assertIn("mesh=init_mesh", engine_source)
        self.assertNotIn("._handle.reshard(True)", worker_source)
        self.assertNotIn("._handle.reshard(True)", engine_source)
        self.assertGreaterEqual(
            worker_source.count("maybe_reshard_fsdp1_root("),
            6,
        )
        self.assertIn("maybe_reshard_fsdp1_root(", engine_source)

    def test_gpu_oracle_checks_global_gradient_and_update(self) -> None:
        source = (
            self.root / "tests" / "fsdp_domain_gradient_oracle.py"
        ).read_text(encoding="utf-8")

        self.assertIn("gradient_coordinate_max_abs_diff", source)
        self.assertIn("accumulation_max_abs_diff", source)
        self.assertIn("parameter_update_max_abs_diff", source)
        self.assertIn("fsdp_process_group_size == world_size", source)

    def test_checkpoint_records_and_validates_effective_topology(self) -> None:
        manager_source = (
            self.root
            / "third_party"
            / "verl"
            / "verl"
            / "utils"
            / "checkpoint"
            / "fsdp_checkpoint_manager.py"
        ).read_text(encoding="utf-8")
        topology_source = (
            self.root
            / "third_party"
            / "verl"
            / "verl"
            / "utils"
            / "checkpoint"
            / "fsdp_checkpoint_topology.py"
        ).read_text(encoding="utf-8")

        self.assertIn("FullStateDictConfig(offload_to_cpu=True", topology_source)
        self.assertIn("FullOptimStateDictConfig(offload_to_cpu=True", topology_source)
        self.assertIn("effective_sharding_strategy", topology_source)
        self.assertIn("fsdp_process_group_size", topology_source)
        self.assertIn("validate_checkpoint_topology(local_path", manager_source)
        self.assertIn(
            "Cross-topology FSDP checkpoint restore is unsupported",
            topology_source,
        )


if __name__ == "__main__":
    unittest.main()
