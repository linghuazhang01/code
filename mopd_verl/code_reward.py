"""Code validation rewards for paper-eval datasets used in MOPD validation."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import subprocess
import uuid
from typing import Any


_DOCKER_RUNNER = """
import os
import sys

source = sys.stdin.read()
exit_process = os._exit
with open(os.devnull, "w", encoding="utf-8") as sink:
    sys.stdin = open(os.devnull, "r", encoding="utf-8")
    sys.stdout = sink
    sys.stderr = sink
    try:
        exec(source, {})
    except BaseException:
        exit_process(1)
exit_process(0)
"""


def _extract_code(completion: str) -> str:
    """Extract code using the same last-Python-block contract as training."""

    return completion.split("```python")[-1].split("```", 1)[0]


def _reliability_guard() -> None:
    try:
        from verl.utils.reward_score.prime_code.testing_util import reliability_guard

        reliability_guard()
    except Exception:
        pass


def _run_assert_case(source: str, result: Any) -> None:
    _reliability_guard()
    namespace: dict[str, Any] = {}
    try:
        signal.alarm(10)
        exec(source, namespace)  # noqa: S102 - benchmark reward execution path.
        signal.alarm(0)
        result.append(True)
    except Exception as exc:
        signal.alarm(0)
        result.append({"error": repr(exc)})


def _remove_docker_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )


def _run_docker_assert_case(source: str) -> tuple[bool, dict[str, Any]]:
    image = os.environ.get("MOPD_CODE_SANDBOX_IMAGE", "verlai/verl:vllm023.dev1")
    container_name = f"mopd-code-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--name",
        container_name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1",
        "--pids-limit=32",
        "--ipc=none",
        "--user=65534:65534",
        "--ulimit=nofile=64:64",
        "--ulimit=nproc=32:32",
        "--ulimit=cpu=10:10",
        "--ulimit=fsize=1048576:1048576",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--env=PYTHONHASHSEED=42",
        "--entrypoint=python",
        image,
        "-I",
        "-c",
        _DOCKER_RUNNER,
    ]
    try:
        completed = subprocess.run(
            command,
            input=source,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        _remove_docker_container(container_name)
        return False, {"error": "timeout", "sandbox": "docker"}
    except FileNotFoundError as exc:
        raise RuntimeError("Docker is required for isolated Code scoring.") from exc

    metadata: dict[str, Any] = {
        "sandbox": "docker",
        "container_image": image,
        "returncode": completed.returncode,
    }
    if completed.returncode == 0:
        metadata["passed"] = True
        return True, metadata
    if completed.returncode in {125, 126, 127}:
        stderr = completed.stderr.strip().replace("\n", " ")[:1000]
        raise RuntimeError(
            "Docker Code sandbox failed before producing a score "
            f"(returncode={completed.returncode}, stderr={stderr!r})."
        )
    metadata["error"] = "assertion_failed" if completed.returncode == 1 else "sandboxed_program_failure"
    return False, metadata


def _assert_score(completion: str, payload: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    code = _extract_code(completion)
    prompt = str(payload.get("prompt", ""))
    assert_case = str(payload.get("assert_case", "")).strip()
    if not assert_case:
        return 0.0, [{"error": "missing assert_case"}]

    candidates = [code]
    if prompt and not code.lstrip().startswith(prompt.lstrip()[:20]):
        candidates.append(prompt + "\n" + code)

    if os.environ.get("MOPD_CODE_SANDBOX") == "docker":
        for source in candidates:
            passed, metadata = _run_docker_assert_case(source + "\n" + assert_case)
            if passed:
                return 1.0, [metadata]
        return 0.0, [metadata]

    for source in candidates:
        manager = multiprocessing.Manager()
        result = manager.list()
        process = multiprocessing.Process(target=_run_assert_case, args=(source + "\n" + assert_case, result))
        process.start()
        process.join(timeout=12)
        if process.is_alive():
            process.kill()
            metadata = [{"error": "timeout"}]
        else:
            metadata = list(result) or [{"error": "empty result"}]
        if metadata and metadata[0] is True:
            return 1.0, [{"passed": True}]
    return 0.0, metadata if isinstance(metadata, list) else [{"error": str(metadata)}]


def _input_output_score(completion: str, payload: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    if os.environ.get("MOPD_CODE_SANDBOX") == "docker":
        raise RuntimeError(
            "LiveCodeBench scoring is disabled in Docker mode because its "
            "input/output scorer has not yet been isolated."
        )
    from verl.utils.reward_score import prime_code

    return prime_code.compute_score(completion, payload, continuous=False)


def compute_score(data_source: str, completion: str, ground_truth: str | dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Compute validation score for EvalPlus/LiveCodeBench-style code datasets."""

    if isinstance(ground_truth, str):
        payload = json.loads(ground_truth)
    else:
        payload = ground_truth

    os.environ.setdefault("PYTHONINTMAXSTRDIGITS", "0")
    if data_source in {"HumanEvalPlus", "MBPPPlus"}:
        return _assert_score(completion, payload)
    if data_source == "LiveCodeBench":
        return _input_output_score(completion, payload)
    return 0.0, [{"error": f"unsupported data_source: {data_source}"}]
