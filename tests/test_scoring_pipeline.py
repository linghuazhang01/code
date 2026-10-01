"""CPU-only regression tests for deferred Code scoring and wave barriers."""
from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from eval.common import EvalResult, EvalSample
from eval.parallel_tasks import run_standard_task
from eval.parallel_worker import run_worker
from eval.scoring_pipeline import ScoringPipeline


def fixture_task(root: Path, name: str = "code", count: int = 2) -> dict:
    output = root / name
    return dict(task_id=name, task_type="standard", dataset=name, domain="code",
                source_start=0, source_end_exclusive=count, num_samples=8,
                generation_seed=42, expected_records=count * 8,
                output_dir=str(output), success_marker=str(output / "SUCCESS"))


def fixture_manifest(tasks: list[dict]) -> dict:
    return dict(schema_version=2, suite="parallel_slurm_eval",
                model=dict(path="model", eval_path="model"),
                execution=dict(batch_size=1, score_code=True, gpu_memory=0.85,
                               max_model_len=18432, max_num_batched_tokens=32768,
                               max_num_seqs=24, enforce_eager=True,
                               enable_chunked_prefill=False,
                               async_code_scoring=dict(workers=2, pending_shards=2)),
                generation=dict(max_new_tokens=16, temperature=1.0, top_p=1.0),
                tasks=tasks, sources=[dict(dataset=t["dataset"], source_file="unused") for t in tasks])


def fake_score(sample: EvalSample, completion: str, *, score_code: bool) -> tuple:
    assert score_code
    score = float(int(completion) % 2 == 0)
    return score, completion, [{"sample": sample.sample_id}]


def fake_generate(llm: object, tokenizer: object, samples: list, **kwargs: object) -> list:
    records = []
    for sample in samples:
        for rollout in range(int(kwargs["num_return_sequences"])):
            result = EvalResult(
                mode="non_thinking", enable_thinking=False, sample_id=sample.sample_id,
                dataset=sample.dataset, ability="code", ground_truth=sample.ground_truth,
                prediction="", score=None, correct=None, prompt_tokens=1,
                generated_tokens=1, thinking_tokens=0, answer_tokens=1, total_tokens=2,
                latency_seconds=0.1, generated_tokens_per_second=10,
                completion_preview=str(rollout), completion=str(rollout),
                rollout_index=rollout, generation_seed=int(kwargs["generation_seed"]),
            )
            if kwargs["score_code"]:
                score, prediction, metadata = fake_score(sample, result.completion, score_code=True)
                result = replace(result, score=score, correct=score == 1.0,
                                 prediction=prediction, reward_metadata=metadata)
            records.append(result)
    return records


def test_deferred_matches_sync_and_preserves_raw(tmp_path: Path) -> None:
    samples = [EvalSample(str(i), "code", "code", [], "tests") for i in range(2)]
    task = fixture_task(tmp_path)
    manifest = fixture_manifest([task])
    with (patch("eval.parallel_tasks.load_eval_samples", return_value=samples),
          patch("eval.parallel_tasks.generate_vllm_batch", side_effect=fake_generate),
          patch("eval.parallel_tasks.score_completion", side_effect=fake_score),
          ThreadPoolExecutor(max_workers=2) as scorers):
        common = dict(manifest=manifest, eval_model_path="model", llm=None,
                      tokenizer=None, source_file=Path("unused"), resume=False)
        finalize = run_standard_task(task=task, scoring_executor=scorers, **common)
        assert callable(finalize)
        assert not Path(task["success_marker"]).exists()
        assert list(tmp_path.glob(".code.worker-*/raw/thinking_eval_samples.jsonl"))
        finalize()
        synchronous = fixture_task(tmp_path, "sync")
        run_standard_task(task=synchronous, **common)
    assert (tmp_path / "code/thinking_eval_samples.jsonl").read_text() == (
        tmp_path / "sync/thinking_eval_samples.jsonl").read_text()
    rows = [json.loads(line) for line in (tmp_path / "code/thinking_eval_samples.jsonl").read_text().splitlines()]
    assert len(rows) == 16
    assert [row["rollout_index"] for row in rows] == list(range(8)) * 2
    assert [row["generation_seed"] for row in rows] == [42] * 8 + [43] * 8
    assert (tmp_path / "code/stage_timings.json").is_file()


def test_scoring_failure_keeps_raw_without_success(tmp_path: Path) -> None:
    task = fixture_task(tmp_path, count=1)
    with (patch("eval.parallel_tasks.load_eval_samples", return_value=[EvalSample("0", "code", "code", [], "tests")]),
          patch("eval.parallel_tasks.generate_vllm_batch", side_effect=fake_generate),
          patch("eval.parallel_tasks.score_completion", side_effect=RuntimeError("scorer failed")),
          ThreadPoolExecutor(max_workers=2) as scorers):
        finalize = run_standard_task(task=task, manifest=fixture_manifest([task]),
                                     eval_model_path="model", llm=None, tokenizer=None,
                                     source_file=Path("unused"), resume=False,
                                     scoring_executor=scorers)
        with pytest.raises(RuntimeError, match="scorer failed"):
            finalize()
    assert not Path(task["output_dir"]).exists()
    assert list(tmp_path.glob(".code.worker-*/raw/thinking_eval_samples.jsonl"))


def test_pipeline_bounds_and_drains() -> None:
    started, release, waiting, acquired = (threading.Event() for _ in range(4))

    def score() -> None:
        started.set()
        assert release.wait(5)

    with ScoringPipeline(2, 1) as pipeline, ThreadPoolExecutor(1) as caller:
        pipeline.submit(score)
        assert started.wait(5)

        def request_room() -> None:
            waiting.set()
            pipeline.make_room()
            acquired.set()

        future = caller.submit(request_room)
        try:
            assert waiting.wait(5)
            assert not acquired.wait(0.05)
        finally:
            release.set()
        future.result(timeout=5)
        assert acquired.is_set()
        assert not pipeline.pending


@pytest.mark.parametrize("fail", [False, True])
def test_worker_overlaps_generation_and_waits_at_wave_boundary(tmp_path: Path, fail: bool) -> None:
    tasks = [fixture_task(tmp_path, name, 1) for name in ("a", "b", "c")]
    manifest = fixture_manifest(tasks)
    manifest["waves"] = [dict(wave_index=0, dataset="first", expected_tasks=2, task_ids=["a", "b"]),
                         dict(wave_index=1, dataset="second", expected_tasks=1, task_ids=["c"])]
    for wave in manifest["waves"]:
        queue = tmp_path / "queue/waves" / f"{wave['wave_index']:04d}_{wave['dataset']}"
        for state in ("pending", "running", "done", "failed"):
            (queue / state).mkdir(parents=True)
        for task_id in wave["task_ids"]:
            (queue / "pending" / f"{task_id}.task").write_text("\t".join(["0", task_id] + ["x"] * 9))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    second_generated = threading.Event()
    first_scored = threading.Event()

    def generate(*, task: dict, **kwargs: object) -> object:
        if task["task_id"] == "b":
            second_generated.set()
        if task["task_id"] == "c":
            assert first_scored.is_set()

        def finalize() -> None:
            if task["task_id"] == "a":
                assert second_generated.wait(5), "Generation did not overlap scoring"
                if fail:
                    raise RuntimeError("intentional scoring failure")
                first_scored.set()
            marker = Path(task["success_marker"])
            marker.parent.mkdir(exist_ok=True)
            marker.touch()
        return finalize

    with (patch("eval.parallel_worker.run_standard_task", side_effect=generate),
          patch("eval.parallel_worker.load_vllm_model", return_value=SimpleNamespace(get_tokenizer=lambda: None)),
          patch("eval.parallel_worker._validate_single_visible_gpu"),
          patch.dict("os.environ", {"MOPD_CODE_SANDBOX": "docker"})):
        status = run_worker(manifest_path=path, eval_model_path="model", worker_id=0, resume=False)
    assert status == int(fail)
    assert Path(tasks[2]["success_marker"]).exists() == (not fail)
    if fail:
        assert list((tmp_path / "queue/waves/0000_first/failed").glob("*.task"))
        assert not (tmp_path / "queue/waves/0000_first/SUCCESS").exists()


def test_pipeline_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        ScoringPipeline(0, 1)
    with pytest.raises(ValueError):
        ScoringPipeline(1, 0)


def test_pipeline_failure_still_finalizes_every_claimed_shard() -> None:
    release = threading.Event()
    finished = []

    def fail() -> None:
        assert release.wait(5)
        raise RuntimeError("first failure")

    with pytest.raises(RuntimeError, match="first failure"):
        with ScoringPipeline(1, 3) as pipeline:
            pipeline.submit(fail)
            pipeline.submit(lambda: finished.append("second"))
            pipeline.submit(lambda: finished.append("third"))
            release.set()
    assert finished == ["second", "third"]


def test_real_deferred_scoring_uses_two_threads(tmp_path: Path) -> None:
    task = fixture_task(tmp_path, count=1)
    pair = threading.Barrier(2)

    def paired_score(*args: object, **kwargs: object) -> tuple:
        pair.wait(timeout=5)
        return fake_score(*args, **kwargs)

    with (patch("eval.parallel_tasks.load_eval_samples", return_value=[EvalSample("0", "code", "code", [], "tests")]),
          patch("eval.parallel_tasks.generate_vllm_batch", side_effect=fake_generate),
          patch("eval.parallel_tasks.score_completion", side_effect=paired_score),
          ThreadPoolExecutor(max_workers=2) as scorers):
        finalize = run_standard_task(task=task, manifest=fixture_manifest([task]),
                                     eval_model_path="model", llm=None, tokenizer=None,
                                     source_file=Path("unused"), resume=False,
                                     scoring_executor=scorers)
        finalize()
    assert Path(task["success_marker"]).exists()


def test_async_refuses_non_docker_backend(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(fixture_manifest([])))
    with patch.dict("os.environ", {"MOPD_CODE_SANDBOX": "native"}):
        with pytest.raises(ValueError, match="requires MOPD_CODE_SANDBOX=docker"):
            run_worker(manifest_path=path, eval_model_path="model", worker_id=0, resume=False)


def test_start_local_dry_run_passes_async_options(tmp_path: Path) -> None:
    code = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(code / "start.sh"), "--eval", "--local", "--model_path", str(tmp_path),
         "--datasets", "humaneval_plus", "--score_code", "--code_scoring_workers", "2",
         "--code_scoring_pending_shards", "2", "--dry_run"],
        capture_output=True, text=True, check=False, cwd=code,
    )
    assert result.returncode == 0, result.stderr
    assert "--code-scoring-workers 2" in result.stdout
    assert "--code-scoring-pending-shards 2" in result.stdout
