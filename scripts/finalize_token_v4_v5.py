"""Verify and import the finalized four-baseline Token V4/V5 revision 3.

Replaces the revision-1 tokenizer-only resolver: unsupported seeds cannot be
retained. Source JSON/CSV are copied byte-for-byte after NPZ support checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

BASELINES = ("1p7b_opd", "1p7b_eopd", "4b_opd", "4b_eopd")
ARTIFACT_SHA256 = "2a6e62db4c0986d4bd4825bc0891cf223fb2b1a955d1bc5404eed2ff42dd22d8"
REVISION_TWO_SHA256 = "c45b5f7b6bd824140922706b2cf75630dce7130d580c346c638a3358833fd97b"
MEMBERSHIP_SHA256 = "7c27427726f985036f2c6b415358508af61b51444782a2fd9cedb8ca15e4d16b"
ANSWER_WORDS = frozenset({"final", "answer", "conclusion"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_answer_word(surface: str) -> bool:
    """Strip leading whitespace and at most one prefix symbol before casefold."""
    core = surface.lstrip()
    if core[:1] in {"(", ".", "[", "\\", "{"}:
        core = core[1:]
    return core.rstrip().casefold() in ANSWER_WORDS


def validate_membership(
    artifact: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Validate strict support, CSV/JSON identity and V5 sibling provenance."""
    meta = artifact.get("meta", {})
    if (
        tuple(meta.get("baselines", ())) != BASELINES
        or set(meta.get("control_class_C_moved_to_structure", {}).get("words", ()))
        != ANSWER_WORDS
    ):
        raise ValueError("Requires finalized four-baseline revision 3 metadata.")
    keep: dict[tuple[str, str], set[int]] = {}
    supported_answers: dict[str, set[int]] = {"math": set(), "code": set()}
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        domain, family, token_id = row["domain"], row["family"], int(row["token_id"])
        key = (domain, family, token_id)
        if key in seen or domain not in {"math", "code"}:
            raise ValueError(f"Duplicate or unsupported membership row: {key}")
        seen.add(key)
        counts = [float(row[f"maxocc_{domain}_{baseline}"]) for baseline in BASELINES]
        if not all(
            np.isfinite(value) and value >= 0 and value.is_integer() for value in counts
        ):
            raise ValueError(f"Invalid maximum occurrence at {key}")
        if min(counts) != float(row["min_maxocc"]):
            raise ValueError(f"Incorrect minimum occurrence at {key}")
        answer_word = _is_answer_word(row["token"])
        if answer_word and min(counts) > 20:
            supported_answers[domain].add(token_id)
        if row["status"] == "keep":
            if min(counts) <= 20:
                raise ValueError(f"Strict four-baseline support >20 failed at {key}")
            if family.startswith("Control_") and (
                answer_word
                or row["control_class"]
                not in ({"A", "B", "D"} if family == "Control_V4" else {"A", "B"})
            ):
                raise ValueError(
                    f"AnswerWord or invalid class retained in Control: {key}"
                )
            keep.setdefault((domain, family), set()).add(token_id)
    summary = {}
    for domain, counts in (("math", (119, 61, 9, 57)), ("code", (153, 72, 45, 64))):
        for version, control_count in (("v4", counts[0]), ("v5", counts[1])):
            entry = artifact[version][domain]
            for category, expected in (
                ("control", control_count),
                ("structure", counts[2]),
            ):
                ids = entry[category]
                family = (
                    "Structure"
                    if category == "structure"
                    else f"Control_{version.upper()}"
                )
                if (
                    ids != sorted(set(ids))
                    or len(ids) != expected
                    or any(t < 0 or t >= 151936 for t in ids)
                ):
                    raise ValueError(f"Invalid {version}/{domain}/{category} IDs")
                if set(ids) != keep.get((domain, family), set()):
                    raise ValueError(
                        f"CSV/JSON mismatch: {version}/{domain}/{category}"
                    )
            if set(entry["control"]) & set(entry["structure"]):
                raise ValueError(f"Control/Structure conflict: {version}/{domain}")
            if not supported_answers[domain] <= set(entry["structure"]):
                raise ValueError(
                    f"Supported AnswerWord missing from Structure: {domain}"
                )
            if any(key.startswith("structure_pending") for key in entry):
                raise ValueError("Pending revision-1 surfaces are not permitted.")
        v4, v5 = artifact["v4"][domain], artifact["v5"][domain]
        intersection = set(v4["control"]) & set(v5["control"])
        siblings = sorted(set(v5["control"]) - set(v4["control"]))
        if v4["structure"] != v5["structure"] or len(intersection) != counts[3]:
            raise ValueError(
                f"Incorrect shared Structure or Control intersection: {domain}"
            )
        if (
            v5.get("control_sibling_added") != siblings
            or len(siblings) != {"math": 4, "code": 8}[domain]
        ):
            raise ValueError(f"V5 sibling provenance mismatch: {domain}")
        summary[domain] = {
            "v4_control": counts[0],
            "v5_control": counts[1],
            "structure": counts[2],
            "control_intersection": counts[3],
            "control_sibling_added": siblings,
        }
    return summary


def verify_source(source: Path) -> dict[str, Any]:
    """Cross-check every CSV count, including rejected tokens, against NPZ."""
    artifact = source / "token_v4_v5.json"
    membership = source / "tables/token_v4_v5_membership.csv"
    with membership.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = validate_membership(
        json.loads(artifact.read_text(encoding="utf-8")), rows
    )
    if _sha256(artifact) != ARTIFACT_SHA256 or _sha256(membership) != MEMBERSHIP_SHA256:
        raise ValueError("Source differs from the approved revision-3 artifact hashes.")
    sources = {
        str(path.relative_to(source)): _sha256(path) for path in (artifact, membership)
    }
    support: dict[str, list[np.ndarray]] = {domain: [] for domain in summary}
    compared = 0
    for baseline in BASELINES:
        path = source / f"tables/four_baseline_maxocc/maxocc_{baseline}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            for domain in summary:
                values = arrays[domain]
                if (
                    values.shape != (151936,)
                    or not np.issubdtype(values.dtype, np.integer)
                    or (values < 0).any()
                ):
                    raise ValueError(
                        f"Invalid full-vocabulary NPZ array: {baseline}/{domain}"
                    )
                support[domain].append(values.copy())
            for row in rows:
                domain, token_id = row["domain"], int(row["token_id"])
                if arrays[domain][token_id] != float(
                    row[f"maxocc_{domain}_{baseline}"]
                ):
                    raise ValueError(
                        f"NPZ/CSV mismatch: {baseline}/{domain}/{token_id}"
                    )
                compared += 1
        sources[str(path.relative_to(source))] = _sha256(path)
    support_counts = {
        domain: int((np.stack(values) > 20).all(axis=0).sum())
        for domain, values in support.items()
    }
    if support_counts != {"math": 3686, "code": 6573}:
        raise ValueError(f"Unexpected full-vocabulary support counts: {support_counts}")
    for name in (
        "four_baseline_maxocc.py",
        "closure_fullvocab.py",
        "build_v4v5_fourbaseline.py",
    ):
        path = source / "scripts" / name
        sources[str(path.relative_to(source))] = _sha256(path)
    return {
        "revision": 3,
        "support_rule": "all four baseline domain maxima > 20",
        "summary": summary,
        "support_counts": support_counts,
        "npz_csv_compared_values": compared,
        "npz_csv_mismatches": 0,
        "sources_sha256": sources,
    }


def install(source: Path, destination: Path, report: dict[str, Any]) -> None:
    """Preserve revision 2 before importing; never modify revision-1 history."""
    previous = destination / "token_v4_v5.json"
    if previous.exists() and _sha256(previous) != ARTIFACT_SHA256:
        if _sha256(previous) != REVISION_TWO_SHA256:
            raise ValueError("Refusing to replace an unknown taxonomy artifact.")
        history = destination / "token_taxonomy_history/revision_2"
        history.mkdir(parents=True, exist_ok=True)
        for name in (
            "token_v4_v5.json",
            "token_v4_v5_membership.csv",
            "token_v4_v5_provenance.json",
        ):
            original, backup = destination / name, history / name
            if backup.exists() and backup.read_bytes() != original.read_bytes():
                raise ValueError(f"Refusing to overwrite a differing backup: {backup}")
            backup.write_bytes(original.read_bytes())
    destination.mkdir(parents=True, exist_ok=True)
    for source_path, name in (
        (source / "token_v4_v5.json", "token_v4_v5.json"),
        (source / "tables/token_v4_v5_membership.csv", "token_v4_v5_membership.csv"),
    ):
        (destination / name).write_bytes(source_path.read_bytes())
    (destination / "token_v4_v5_provenance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Approved four-baseline source directory"
    )
    parser.add_argument(
        "--install",
        type=Path,
        metavar="DIRECTORY",
        help="Import after verification; defaults to verify-only",
    )
    args = parser.parse_args()
    report = verify_source(args.source)
    if args.install:
        install(args.source, args.install, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
