from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mopd_verl.domain_gradient import token_logging
from mopd_verl.domain_gradient.token_logging import (
    LocalTokenCandidate,
    append_token_vocab_vectors_jsonl,
)
from mopd_verl.domain_gradient.token_selection import RankedToken


class TokenGradientLoggingTests(unittest.TestCase):
    def test_logging_does_not_gather_per_occurrence_python_objects(
        self,
    ) -> None:
        source = Path(token_logging.__file__).read_text(encoding="utf-8")

        self.assertNotIn("all_gather_object", source)
        self.assertIn("torch.distributed.all_reduce", source)

    def test_duplicate_token_ids_accumulate_into_vocab_vectors(self) -> None:
        candidates = (
            LocalTokenCandidate(
                micro_batch_index=0,
                sample_index=0,
                token_index=0,
                token_id=42,
                configured_loss=-0.25,
                loss_abs=0.25,
            ),
            LocalTokenCandidate(
                micro_batch_index=0,
                sample_index=0,
                token_index=1,
                token_id=42,
                configured_loss=0.50,
                loss_abs=0.50,
            ),
            LocalTokenCandidate(
                micro_batch_index=0,
                sample_index=0,
                token_index=2,
                token_id=7,
                configured_loss=1.00,
                loss_abs=1.00,
            ),
        )
        selected = tuple(
            RankedToken(
                owner_rank=0,
                owner_index=index,
                loss_abs=candidate.loss_abs,
            )
            for index, candidate in enumerate(candidates)
        )

        with tempfile.TemporaryDirectory() as directory:
            append_token_vocab_vectors_jsonl(
                output_dir=directory,
                step=5,
                configured_vocab_size=50,
                candidates_by_domain={"math": candidates},
                selections_by_domain={
                    "math": {
                        "tail": (selected[0],),
                        "top_p": selected,
                    }
                },
            )
            output_path = (
                Path(directory) / "token_gradient_vocab_vectors.jsonl"
            )
            rows = [
                json.loads(line)
                for line in output_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        self.assertEqual(len(rows), 2)
        top_p = next(row for row in rows if row["selection"] == "top_p")
        self.assertEqual(top_p["step"], 5)
        self.assertEqual(top_p["domain"], "math")
        self.assertEqual(top_p["vector_value"], "cumulative_occurrence_count")
        self.assertEqual(top_p["selected_token_count"], 3)
        self.assertEqual(top_p["nonzero_token_ids"], [7, 42])
        self.assertEqual(top_p["token_count_vector_vocab"][7], 1)
        self.assertEqual(top_p["token_count_vector_vocab"][42], 2)
        self.assertAlmostEqual(
            top_p["configured_token_loss_sum_vector_vocab"][42],
            0.25,
        )
        self.assertAlmostEqual(
            top_p["configured_token_loss_abs_sum_vector_vocab"][42],
            0.75,
        )

    def test_vocab_size_falls_back_to_observed_max_token_id(self) -> None:
        candidate = LocalTokenCandidate(
            micro_batch_index=0,
            sample_index=0,
            token_index=0,
            token_id=3,
            configured_loss=1.0,
            loss_abs=1.0,
        )
        selected = RankedToken(
            owner_rank=0,
            owner_index=0,
            loss_abs=1.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            append_token_vocab_vectors_jsonl(
                output_dir=directory,
                step=1,
                configured_vocab_size=None,
                candidates_by_domain={"code": (candidate,)},
                selections_by_domain={"code": {"top_p": (selected,)}},
            )
            output_path = (
                Path(directory) / "token_gradient_vocab_vectors.jsonl"
            )
            row = json.loads(
                output_path.read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(row["vocab_size"], 4)
        self.assertEqual(row["token_count_vector_vocab"], [0, 0, 0, 1])

    def test_distributed_sum_aggregates_rank_local_vectors(self) -> None:
        candidate = LocalTokenCandidate(
            micro_batch_index=0,
            sample_index=0,
            token_index=0,
            token_id=2,
            configured_loss=-0.5,
            loss_abs=0.5,
        )
        selected = RankedToken(
            owner_rank=0,
            owner_index=0,
            loss_abs=0.5,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                token_logging,
                "_distributed_context",
                return_value=(True, 0),
            ),
            patch.object(
                token_logging,
                "_collective_device",
                return_value=torch.device("cpu"),
            ),
            patch.object(
                token_logging.torch.distributed,
                "all_reduce",
                side_effect=lambda tensor, *args, **kwargs: tensor.mul_(2),
            ),
        ):
            append_token_vocab_vectors_jsonl(
                output_dir=directory,
                step=2,
                configured_vocab_size=4,
                candidates_by_domain={"science": (candidate,)},
                selections_by_domain={
                    "science": {"top_p": (selected,)}
                },
            )
            output_path = (
                Path(directory) / "token_gradient_vocab_vectors.jsonl"
            )
            row = json.loads(
                output_path.read_text(encoding="utf-8").splitlines()[0]
            )

        self.assertEqual(row["token_count_vector_vocab"], [0, 0, 2, 0])
        self.assertEqual(
            row["configured_token_loss_sum_vector_vocab"],
            [0.0, 0.0, -1.0, 0.0],
        )
        self.assertEqual(
            row["configured_token_loss_abs_sum_vector_vocab"],
            [0.0, 0.0, 1.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
