"""Official scoring must be included in canonical local completion."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("fail", [False, True])
def test_finalizer_publishes_success_only_after_official_scoring(tmp_path: Path, fail: bool) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts/finalize_local_standard_eval.sh"
    shutil.copy2(source, tmp_path / source.name)
    worker = tmp_path / "slurm_evalplus_rescore_worker.sh"
    worker.write_text(
        'set -eu\n'
        'test "$(basename "$2")" = prompt_response_records.jsonl\n'
        'test -z "$CUDA_VISIBLE_DEVICES"\n'
        + ('exit 7\n' if fail else 'mkdir -p "$3/humaneval" "$3/mbpp"\ntouch "$3/SUCCESS" "$3/humaneval/SUCCESS" "$3/mbpp/SUCCESS"\n')
    )
    suite = tmp_path / "suite"
    (suite / "model/code").mkdir(parents=True)
    (suite / "logs").mkdir()
    (suite / "model/code/prompt_response_records.jsonl").touch()
    (suite / "MERGE_SUCCESS").touch()
    (suite / "suite_manifest.json").write_text(json.dumps({"model": {"label": "model"}}))
    result = subprocess.run(["bash", str(tmp_path / source.name), str(suite)],
                            env={**os.environ, "PYTHON": sys.executable}, capture_output=True)
    assert (result.returncode != 0) == fail
    assert (suite / "MERGE_SUCCESS").is_file()
    assert (suite / "SUCCESS").exists() == (not fail)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("standard", [False, True])
def test_start_scores_custom_code_suite_and_propagates_resume(
    tmp_path: Path, fail: bool, standard: bool,
) -> None:
    """Exercise the real start/launcher/finalizer chain with CPU-only test workers."""
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    for filename in ["start.sh", "scripts/run_local_math_parallel_eval.sh",
                     "scripts/finalize_local_standard_eval.sh"]:
        shutil.copy2(source / filename, repo / filename)
    runtime = repo / ".runtime/evalplus-gopd-37371a4c31ad7947746200d234161769191f4748/venv/bin"
    runtime.mkdir(parents=True)
    (runtime / "python").symlink_to(sys.executable)
    fake = tmp_path / "bin"
    fake.mkdir()
    router = fake / "route.py"
    router.write_text(textwrap.dedent('''\
        import json, sys
        from pathlib import Path
        args = sys.argv[1:]
        def value(flag):
            return args[args.index(flag) + 1]
        if args[0] == 'eval.parallel_worker':
            assert '--resume' in args
        elif args[1] == 'plan':
            assert '--resume' in args
            root = Path(value('--suite-root'))
            (root / 'model/code').mkdir(parents=True, exist_ok=True)
            (root / 'model/code/prompt_response_records.jsonl').touch()
            (root / 'suite_manifest.json').write_text(json.dumps({'model': {'label': 'model'}}))
        elif args[1] == 'merge':
            assert '--defer-completion' in args
            Path(value('--manifest')).with_name('MERGE_SUCCESS').touch()
        else:
            raise AssertionError(args)
    '''))
    python = fake / "python"
    python.write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        'if [[ "${1:-}" == -m ]]; then\n'
        f'  shift; exec "{sys.executable}" "{router}" "$@"\n'
        'fi\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    docker = fake / "docker"
    docker.write_text('#!/usr/bin/env bash\necho test-image-id\n')
    docker.chmod(0o755)
    (scripts / "slurm_evalplus_rescore_worker.sh").write_text(
        'set -eu\ntest -z "$CUDA_VISIBLE_DEVICES"\n'
        + ('exit 7\n' if fail else
           'mkdir -p "$3/humaneval" "$3/mbpp"\n'
           'touch "$3/SUCCESS" "$3/humaneval/SUCCESS" "$3/mbpp/SUCCESS"\n')
    )
    suite = tmp_path / "output/test"
    suite.mkdir(parents=True)
    datasets = 'aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6'
    command = ['bash', str(repo / 'start.sh'), '--eval', '--local',
               '--model_path', str(tmp_path), '--output_root', str(suite.parent),
               '--run_tag', 'test', '--gpus', '2', '--gpu_ids', '0,1',
               '--datasets', datasets, '--gopd_dir', str(tmp_path), '--score_code', '--resume']
    if standard:
        command.append('--standard_protocol')
    result = subprocess.run(command, env={**os.environ, 'PYTHON': str(python),
                                         'PATH': f'{fake}:{os.environ["PATH"]}'},
                            capture_output=True, text=True)
    assert (result.returncode != 0) == fail, result.stderr
    assert (suite / 'MERGE_SUCCESS').exists()
    assert (suite / 'SUCCESS').exists() == (not fail)
    assert (suite / 'COMPLETED_AT_UTC').exists() == (not fail)
    manifest = json.loads((suite / 'suite_manifest.json').read_text())
    assert ('official_evalplus' in manifest) == (not fail)
