from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from mopd_verl.config_profiles import (
    ConfigReference,
    list_config_profiles,
    load_raw_config,
)


class ConfigProfileTests(unittest.TestCase):
    def _write_yaml(
        self,
        directory: str,
        filename: str,
        value: object,
    ) -> Path:
        path = Path(directory) / filename
        path.write_text(
            yaml.safe_dump(value, sort_keys=False),
            encoding="utf-8",
        )
        return path

    def test_regular_yaml_remains_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_yaml(
                temp_dir,
                "regular.yaml",
                {"data": {"batch": 12}, "items": [1, 2]},
            )

            reference = ConfigReference.parse(path)
            resolved = load_raw_config(path)

        self.assertEqual(reference.path, path)
        self.assertIsNone(reference.profile)
        self.assertEqual(
            resolved,
            {"data": {"batch": 12}, "items": [1, 2]},
        )

    def test_matrix_profile_deep_merges_dicts_and_replaces_lists(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_yaml(
                temp_dir,
                "matrix.yaml",
                {
                    "profile_matrix": {
                        "version": 1,
                        "base": {
                            "actor": {
                                "fsdp_size": 1,
                                "nested": {"left": 1, "right": 2},
                            },
                            "tokens": [10, 20],
                        },
                        "profiles": {
                            "fsdp2": {
                                "actor": {
                                    "fsdp_size": 2,
                                    "nested": {"right": 3},
                                },
                                "tokens": [30],
                            }
                        },
                    }
                },
            )

            resolved = load_raw_config(f"{path}::fsdp2")
            resolved_from_path = load_raw_config(
                Path(f"{path}::fsdp2")
            )

        self.assertEqual(
            resolved,
            {
                "actor": {
                    "fsdp_size": 2,
                    "nested": {"left": 1, "right": 3},
                },
                "tokens": [30],
            },
        )
        self.assertEqual(resolved_from_path, resolved)

    def test_matrix_requires_known_explicit_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_yaml(
                temp_dir,
                "matrix.yaml",
                {
                    "profile_matrix": {
                        "version": 1,
                        "base": {"value": 1},
                        "profiles": {
                            "alpha": {},
                            "beta": {"value": 2},
                        },
                    }
                },
            )

            with self.assertRaisesRegex(
                ValueError,
                "requires an explicit profile",
            ):
                load_raw_config(path)
            with self.assertRaisesRegex(
                ValueError,
                "Unknown config profile 'missing'.*alpha, beta",
            ):
                load_raw_config(f"{path}::missing")
            self.assertEqual(
                list_config_profiles(path),
                ("alpha", "beta"),
            )

    def test_regular_yaml_rejects_profile_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_yaml(
                temp_dir,
                "regular.yaml",
                {"value": 1},
            )

            with self.assertRaisesRegex(
                ValueError,
                "does not define a profile matrix",
            ):
                load_raw_config(f"{path}::unexpected")

    def test_reference_rejects_empty_profile(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "profile name must be non-empty",
        ):
            ConfigReference.parse("matrix.yaml::")
        with self.assertRaisesRegex(
            ValueError,
            "may contain only letters",
        ):
            ConfigReference.parse("matrix.yaml::../escape")

    def test_invalid_matrix_shapes_fail_before_resolution(self) -> None:
        cases: list[tuple[str, object, str]] = [
            (
                "unsupported-version",
                {
                    "profile_matrix": {
                        "version": 2,
                        "base": {},
                        "profiles": {"valid": {}},
                    }
                },
                "Unsupported profile matrix version",
            ),
            (
                "sibling-key",
                {
                    "profile_matrix": {
                        "version": 1,
                        "base": {},
                        "profiles": {"valid": {}},
                    },
                    "trainer": {},
                },
                "cannot define sibling top-level keys",
            ),
            (
                "missing-base",
                {
                    "profile_matrix": {
                        "version": 1,
                        "profiles": {"valid": {}},
                    }
                },
                "profile_matrix.base",
            ),
            (
                "invalid-profiles",
                {
                    "profile_matrix": {
                        "version": 1,
                        "base": {},
                        "profiles": [],
                    }
                },
                "profile_matrix.profiles",
            ),
            (
                "empty-profiles",
                {
                    "profile_matrix": {
                        "version": 1,
                        "base": {},
                        "profiles": {},
                    }
                },
                "at least one profile",
            ),
            (
                "invalid-profile-name",
                {
                    "profile_matrix": {
                        "version": 1,
                        "base": {},
                        "profiles": {"../escape": {}},
                    }
                },
                "names may contain only letters",
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, value, expected_error in cases:
                with self.subTest(name=name):
                    path = self._write_yaml(
                        temp_dir,
                        f"{name}.yaml",
                        value,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        expected_error,
                    ):
                        load_raw_config(f"{path}::valid")

    def test_test_grad_directory_contains_canonical_yaml_files(self) -> None:
        config_dir = (
            Path(__file__).resolve().parents[1] / "test_grad_configs"
        )
        self.assertEqual(
            {path.name for path in config_dir.glob("*.yaml")},
            {
                (
                    "mopd_dynamic_budget_qwen0p6b_8b_aw2_fsdp2_"
                    "b16_4step_3gpu_smoke.yaml"
                ),
                "mopd_grad_reliability_qwen0p6b_8b_matrix.yaml",
            },
        )

    def test_all_test_grad_profiles_use_distinct_0p6b_student_8b_teacher(
        self,
    ) -> None:
        config_dir = (
            Path(__file__).resolve().parents[1] / "test_grad_configs"
        )
        expected_student = "/root/autodl-tmp/models/Qwen3-0.6B"
        expected_teacher = "/root/autodl-tmp/models/Qwen3-8B"

        for path in sorted(config_dir.glob("*.yaml")):
            profiles = list_config_profiles(path)
            references = (
                [f"{path}::{profile}" for profile in profiles]
                if profiles
                else [str(path)]
            )
            for reference in references:
                with self.subTest(config=reference):
                    model = load_raw_config(reference)["model"]
                    domain_teachers = set(
                        model.get("domain_teacher_paths", {}).values()
                    )
                    configured_teachers = {
                        path
                        for path in (
                            model.get("math_teacher_path"),
                            model.get("code_teacher_path"),
                            *domain_teachers,
                        )
                        if path is not None
                    }

                    self.assertEqual(
                        model["student_path"],
                        expected_student,
                    )
                    self.assertEqual(
                        configured_teachers,
                        {expected_teacher},
                    )
                    self.assertNotIn(
                        model["student_path"],
                        configured_teachers,
                    )


if __name__ == "__main__":
    unittest.main()
