"""Resolve ordinary YAML configs and named-profile config matrices."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROFILE_SEPARATOR = "::"
PROFILE_MATRIX_KEY = "profile_matrix"
SUPPORTED_MATRIX_VERSION = 1


def _valid_profile_name(value: str) -> bool:
    return bool(value) and all(
        character.isalnum() or character in {"_", "-", "."}
        for character in value
    )


@dataclass(frozen=True)
class ConfigReference:
    """A physical YAML path with an optional named-profile selector."""

    path: Path
    profile: str | None = None

    @classmethod
    def parse(cls, value: str | Path) -> ConfigReference:
        raw = str(value)
        if PROFILE_SEPARATOR not in raw:
            return cls(path=Path(raw))
        path_text, profile = raw.rsplit(PROFILE_SEPARATOR, 1)
        if not profile:
            raise ValueError("Config profile name must be non-empty.")
        if not _valid_profile_name(profile):
            raise ValueError(
                "Config profile name may contain only letters, numbers, "
                "underscores, hyphens, and dots."
            )
        if not path_text:
            raise ValueError("Config path must be non-empty.")
        return cls(path=Path(path_text), profile=profile)

    def as_string(self) -> str:
        if self.profile is None:
            return str(self.path)
        return f"{self.path}{PROFILE_SEPARATOR}{self.profile}"


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{label}' to be a mapping.")
    return value


def _deep_merge(
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, overlay_value in overlay.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(
            overlay_value,
            dict,
        ):
            result[key] = _deep_merge(base_value, overlay_value)
        else:
            result[key] = deepcopy(overlay_value)
    return result


def _read_yaml_root(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(raw, "root")


def _matrix_parts(
    root: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if PROFILE_MATRIX_KEY not in root:
        return None
    extra_keys = set(root) - {PROFILE_MATRIX_KEY}
    if extra_keys:
        rendered = ", ".join(sorted(str(key) for key in extra_keys))
        raise ValueError(
            "Profile matrix YAML cannot define sibling top-level keys: "
            f"{rendered}."
        )
    matrix = _mapping(root[PROFILE_MATRIX_KEY], PROFILE_MATRIX_KEY)
    version = matrix.get("version")
    if version != SUPPORTED_MATRIX_VERSION:
        raise ValueError(
            "Unsupported profile matrix version "
            f"{version!r}; expected {SUPPORTED_MATRIX_VERSION}."
        )
    base = _mapping(matrix.get("base"), f"{PROFILE_MATRIX_KEY}.base")
    profiles = _mapping(
        matrix.get("profiles"),
        f"{PROFILE_MATRIX_KEY}.profiles",
    )
    if not profiles:
        raise ValueError("Profile matrix must define at least one profile.")
    for profile_name, profile_value in profiles.items():
        if not isinstance(profile_name, str) or not _valid_profile_name(
            profile_name
        ):
            raise ValueError(
                "Profile matrix names may contain only letters, numbers, "
                "underscores, hyphens, and dots."
            )
        _mapping(
            profile_value,
            f"{PROFILE_MATRIX_KEY}.profiles.{profile_name}",
        )
    return base, profiles


def list_config_profiles(path: str | Path) -> tuple[str, ...]:
    """Return named profiles in declaration order, or an empty tuple."""

    reference = ConfigReference.parse(path)
    root = _read_yaml_root(reference.path)
    matrix_parts = _matrix_parts(root)
    if matrix_parts is None:
        return ()
    _, profiles = matrix_parts
    return tuple(profiles)


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Resolve a config reference into a standalone raw config mapping."""

    reference = ConfigReference.parse(path)
    root = _read_yaml_root(reference.path)
    matrix_parts = _matrix_parts(root)
    if matrix_parts is None:
        if reference.profile is not None:
            raise ValueError(
                f"Config '{reference.path}' does not define a profile "
                "matrix."
            )
        return deepcopy(root)

    base, profiles = matrix_parts
    if reference.profile is None:
        available = ", ".join(profiles)
        raise ValueError(
            f"Config matrix '{reference.path}' requires an explicit profile "
            f"using '::profile'. Available profiles: {available}."
        )
    if reference.profile not in profiles:
        available = ", ".join(profiles)
        raise ValueError(
            f"Unknown config profile '{reference.profile}' for "
            f"'{reference.path}'. Available profiles: {available}."
        )
    overlay = _mapping(
        profiles[reference.profile],
        f"{PROFILE_MATRIX_KEY}.profiles.{reference.profile}",
    )
    return _deep_merge(base, overlay)
