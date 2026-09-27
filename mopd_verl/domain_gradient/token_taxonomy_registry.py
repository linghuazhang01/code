"""Version-scoped registry for the frozen Token V4/V5 taxonomy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

TOKEN_TAXONOMY_ARTIFACT_SHA256 = (
    "2a6e62db4c0986d4bd4825bc0891cf223fb2b1a955d1bc5404eed2ff42dd22d8"
)
TOKEN_TAXONOMY_REVISION = 3
TOKEN_TAXONOMY_VERSIONS = frozenset({"token_v4", "token_v5"})


@dataclass(frozen=True)
class DomainTokenTaxonomy:
    """One domain's disjoint Control and Structure token sets."""

    control: tuple[int, ...]
    structure: tuple[int, ...]
    control_sibling_added: tuple[int, ...] = ()


def normalize_token_taxonomy_version(value: str) -> str:
    """Normalize public V4/V5 aliases while preserving the legacy default."""

    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {"v4": "token_v4", "v5": "token_v5"}
    return aliases.get(normalized, normalized)


def _artifact_path() -> Path:
    return Path(__file__).with_name("token_v4_v5.json")


def _load_artifact() -> dict[str, object]:
    raw = _artifact_path().read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TOKEN_TAXONOMY_ARTIFACT_SHA256:
        raise ValueError(
            "Token V4/V5 artifact SHA256 mismatch: "
            f"expected {TOKEN_TAXONOMY_ARTIFACT_SHA256}, got {digest}."
        )
    return json.loads(raw)


def token_taxonomy(version: str) -> dict[str, DomainTokenTaxonomy]:
    """Load one frozen taxonomy and validate its fundamental invariants."""

    normalized = normalize_token_taxonomy_version(version)
    if normalized not in TOKEN_TAXONOMY_VERSIONS:
        raise ValueError(f"Unsupported token taxonomy version: {version!r}")
    artifact = _load_artifact()
    raw = artifact[normalized.removeprefix("token_")]
    result: dict[str, DomainTokenTaxonomy] = {}
    for domain in ("math", "code"):
        entry = raw[domain]
        control = tuple(sorted({int(token_id) for token_id in entry["control"]}))
        structure = tuple(sorted({int(token_id) for token_id in entry["structure"]}))
        if not control or not structure or set(control).intersection(structure):
            raise ValueError(
                f"Invalid Control/Structure membership for {normalized}/{domain}."
            )
        siblings = tuple(int(value) for value in entry.get("control_sibling_added", ()))
        v4_control = set(artifact["v4"][domain]["control"])
        if set(siblings) != set(control) - v4_control:
            raise ValueError(
                f"Invalid V5 sibling provenance for {normalized}/{domain}."
            )
        result[domain] = DomainTokenTaxonomy(control, structure, siblings)
    return result


def taxonomy_counts(version: str) -> dict[str, tuple[int, int]]:
    """Return deterministic Control/Structure counts for diagnostics/tests."""

    return {
        domain: (len(value.control), len(value.structure))
        for domain, value in token_taxonomy(version).items()
    }
