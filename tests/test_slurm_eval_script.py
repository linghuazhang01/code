from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPT = CODE_DIR / "slurm_eval.sh"
WORKER = CODE_DIR / "scripts/slurm_eval_worker.sh"


def _make_hf_model(root: Path, name: str) -> Path:
    model_dir = root / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"test")
    return model_dir


class SlurmEvalScriptTest(unittest.TestCase):
    def test_dry_run_requests_one_gpu_and_forwards_repeated_model_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _make_hf_model(root, "first")
            second = _make_hf_model(root, "second")

            completed = subprocess.run(
                [
                    str(SCRIPT),
                    "--model_path",
                    str(first),
                    "--model_path",
                    str(second),
                    "--run_tag",
                    "test-run",
                    "--dry_run",
                ],
                cwd=CODE_DIR,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--gpus=1", completed.stdout)
        self.assertEqual(completed.stdout.count("--model_path"), 2)
        self.assertIn(str(first), completed.stdout)
        self.assertIn(str(second), completed.stdout)

    def test_submit_uses_exactly_one_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = _make_hf_model(root, "model")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            args_path = root / "sbatch-args.txt"
            fake_sbatch = bin_dir / "sbatch"
            fake_sbatch.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"${SBATCH_ARGS_PATH}\"\n"
                "printf '321\\n'\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
            environment["SBATCH_ARGS_PATH"] = str(args_path)

            completed = subprocess.run(
                [str(SCRIPT), "--model_path", str(model), "--run_tag", "submit-test"],
                cwd=CODE_DIR,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            submitted_args = args_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("job_id=321", completed.stdout)
        self.assertEqual(submitted_args.count("--gpus=1"), 1)
        self.assertNotIn("--gpus=2", submitted_args)
        self.assertIn("--mem=24G", submitted_args)

    def test_missing_model_path_is_rejected_before_submission(self) -> None:
        completed = subprocess.run(
            [str(SCRIPT), "--model_path", "/definitely/missing/model"],
            cwd=CODE_DIR,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("model path does not exist", completed.stderr)

    def test_invalid_max_samples_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = _make_hf_model(Path(temp_dir), "model")
            completed = subprocess.run(
                [str(SCRIPT), "--model_path", str(model), "--max_samples", "0"],
                cwd=CODE_DIR,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("must be a positive integer", completed.stderr)

    def test_resume_starts_an_untouched_second_model_without_resume_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = root / "code"
            scripts_dir = fixture / "scripts"
            scripts_dir.mkdir(parents=True)
            worker = scripts_dir / "slurm_eval_worker.sh"
            shutil.copy2(WORKER, worker)
            worker.chmod(0o755)

            calls_path = root / "eval-calls.txt"
            fake_eval = scripts_dir / "run_local_eval.sh"
            fake_eval.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%q ' \"$@\" >> \"${EVAL_CALLS_PATH}\"\n"
                "printf '\\n' >> \"${EVAL_CALLS_PATH}\"\n",
                encoding="utf-8",
            )
            fake_eval.chmod(0o644)
            fake_python = root / "python"
            fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)

            first = _make_hf_model(root, "first")
            second = _make_hf_model(root, "second")
            output_root = root / "output"
            first_output = output_root / "resume-test" / "first" / "math"
            first_output.mkdir(parents=True)
            (first_output / "eval_run_config.json").write_text("{}\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "5",
                    "EVAL_CALLS_PATH": str(calls_path),
                    "SLURM_EVAL_PYTHON": str(fake_python),
                    "SLURM_JOB_ID": "999",
                    "SLURM_SUBMIT_DIR": str(fixture),
                }
            )

            completed = subprocess.run(
                [
                    str(worker),
                    "--datasets",
                    "aime24",
                    "--output_root",
                    str(output_root),
                    "--run_tag",
                    "resume-test",
                    "--resume",
                    "--no_score_code",
                    "--model_path",
                    str(first),
                    "--model_path",
                    str(second),
                ],
                cwd=fixture,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            calls = calls_path.read_text(encoding="utf-8").splitlines()
            empty_resume_file_created = (
                first_output / "thinking_eval_samples.jsonl"
            ).is_file()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(calls), 2)
        self.assertIn("--resume", calls[0])
        self.assertNotIn("--resume", calls[1])
        self.assertIn("--temperature 1.0", calls[0])
        self.assertIn("--top-p 1.0", calls[0])
        self.assertIn("--max-new-tokens 16384", calls[0])
        self.assertIn("--num-samples 32", calls[0])
        self.assertTrue(empty_resume_file_created)

    def test_worker_splits_domain_sampling_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixture = root / "code"
            scripts_dir = fixture / "scripts"
            scripts_dir.mkdir(parents=True)
            worker = scripts_dir / "slurm_eval_worker.sh"
            shutil.copy2(WORKER, worker)
            worker.chmod(0o755)
            calls_path = root / "eval-calls.txt"
            fake_eval = scripts_dir / "run_local_eval.sh"
            fake_eval.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%q ' \"$@\" >> \"${EVAL_CALLS_PATH}\"\n"
                "printf '\\n' >> \"${EVAL_CALLS_PATH}\"\n",
                encoding="utf-8",
            )
            fake_python = root / "python"
            fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)
            model = _make_hf_model(root, "model")
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": "5",
                    "EVAL_CALLS_PATH": str(calls_path),
                    "SLURM_EVAL_PYTHON": str(fake_python),
                    "SLURM_JOB_ID": "1000",
                    "SLURM_SUBMIT_DIR": str(fixture),
                }
            )

            completed = subprocess.run(
                [
                    str(worker),
                    "--datasets",
                    "aime24,humaneval_plus,gpqa_diamond",
                    "--output_root",
                    str(root / "output"),
                    "--run_tag",
                    "protocol-test",
                    "--sample_offset",
                    "3",
                    "--max_samples",
                    "5",
                    "--no_score_code",
                    "--model_path",
                    str(model),
                ],
                cwd=fixture,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            calls = calls_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(calls), 3)
        self.assertIn("--datasets aime24", calls[0])
        self.assertIn("--num-samples 32", calls[0])
        self.assertIn("--datasets humaneval_plus", calls[1])
        self.assertIn("--num-samples 4", calls[1])
        self.assertIn("--datasets gpqa_diamond", calls[2])
        self.assertIn("--num-samples 1", calls[2])
        for call in calls:
            self.assertIn("--temperature 1.0", call)
            self.assertIn("--top-p 1.0", call)
            self.assertIn("--max-new-tokens 16384", call)
            self.assertIn("--sample-offset 3", call)
            self.assertIn("--max-samples 5", call)


if __name__ == "__main__":
    unittest.main()
