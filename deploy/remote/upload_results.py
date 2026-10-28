"""Upload finished evaluation suites to a Hugging Face repo (always huggingface.co, never the read-only mirror).

    upload_results.py --repo icemoon28/opd-eval-results [--repo-type dataset] [--prefix eval_results] SUITE_DIR...

Only suites with COMPLETED_AT_UTC are uploaded. Per-shard scratch files and the EvalPlus problem cache are left out
(they are regenerated); prompts, generations, merged results and official scores are kept. The token comes from
HF_TOKEN (put it in ~/.config/opd-private/opd.env on the machine that uploads).
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

IGNORE = ["shards/*", "shards/**", "**/cache/**", "*.lock"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--prefix", default="eval_results")
    p.add_argument("suites", nargs="+", type=Path)
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; nothing uploaded", file=sys.stderr)
        return 2
    api = HfApi(endpoint="https://huggingface.co", token=token)
    api.create_repo(a.repo, repo_type=a.repo_type, private=True, exist_ok=True)
    failed = 0
    for suite in a.suites:
        if not (suite / "COMPLETED_AT_UTC").exists():
            print(f"skip (not complete): {suite}")
            continue
        path_in_repo = f"{a.prefix.strip('/')}/{suite.name}"
        try:
            api.upload_folder(repo_id=a.repo, repo_type=a.repo_type, folder_path=suite, path_in_repo=path_in_repo,
                              ignore_patterns=IGNORE, commit_message=f"Add evaluation suite {suite.name}")
            print(f"uploaded {suite} -> {a.repo}/{path_in_repo}")
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"FAILED {suite}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
