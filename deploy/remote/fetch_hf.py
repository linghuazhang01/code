"""Download a Hugging Face model repo, or one folder of a repo, and check every file against the hub.

    fetch_hf.py model  <repo_id> <dest_dir> [--revision REV]
    fetch_hf.py prefix <repo_id> <path_in_repo> <root>          # e.g. a global_step_60 folder of icemoon28/opd-checkpoints

Sizes are compared for every file and SHA-256 for every LFS file; HF_PROVENANCE.json records the resolved
revision. A folder whose provenance already matches the hub revision and file list is skipped, so reruns resume.
Honours HF_ENDPOINT (mirror) and the proxy variables set by common.sh.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

if importlib.util.find_spec("hf_transfer") is not None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(repo: str, revision: str | None, prefix: str | None, local_dir: Path, target: Path) -> None:
    api = HfApi()
    rev = api.repo_info(repo, revision=revision).sha
    remote = [i for i in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, expand=True, revision=rev)
              if getattr(i, "size", None) is not None]
    if not any(Path(i.path).name.endswith(".safetensors") for i in remote):
        raise SystemExit(f"no *.safetensors under {repo}/{prefix or ''}@{rev}")
    prov = target / "HF_PROVENANCE.json"
    if prov.exists():
        old = json.loads(prov.read_text())
        if old.get("revision") == rev and {f["path"] for f in old.get("files", [])} == {i.path for i in remote}:
            print(f"up to date: {target} ({repo}@{rev[:8]})")
            return
    patterns = [f"{prefix}/*"] if prefix else None
    snapshot_download(repo, revision=rev, allow_patterns=patterns, local_dir=local_dir, max_workers=8)
    files = []
    for i in remote:
        local = local_dir / i.path
        if local.stat().st_size != i.size:
            raise SystemExit(f"size mismatch {i.path}: {local.stat().st_size} != {i.size}")
        entry = {"path": i.path, "size": i.size}
        if i.lfs is not None:
            digest = sha256(local)
            if digest != i.lfs.sha256:
                raise SystemExit(f"sha256 mismatch {i.path}")
            entry["sha256"] = digest
        files.append(entry)
    prov.write_text(json.dumps({
        "time": dt.datetime.now(dt.timezone.utc).isoformat(), "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        "repo": repo, "revision": rev, "path_in_repo": prefix, "local_path": str(target.resolve()), "files": files,
    }, indent=2) + "\n")
    print(f"downloaded and verified: {target} ({len(files)} files, {repo}@{rev[:8]})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="kind", required=True)
    m = sub.add_parser("model")
    m.add_argument("repo")
    m.add_argument("dest", type=Path)
    m.add_argument("--revision")
    f = sub.add_parser("prefix")
    f.add_argument("repo")
    f.add_argument("path_in_repo")
    f.add_argument("root", type=Path)
    a = p.parse_args()
    if a.kind == "model":
        a.dest.mkdir(parents=True, exist_ok=True)
        fetch(a.repo, a.revision, None, a.dest, a.dest)
    else:
        prefix = a.path_in_repo.strip("/")
        (a.root / prefix).mkdir(parents=True, exist_ok=True)
        fetch(a.repo, None, prefix, a.root, a.root / prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
