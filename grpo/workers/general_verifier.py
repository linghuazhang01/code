"""General-Reasoner model-based verifier reward worker."""

from __future__ import annotations

import logging
import os
import re

import torch
from tensordict import TensorDict
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))

VERIFIER_PROMPT_TEMPLATE = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
    'If the student\'s answer is correct, output "Final Decision: Yes". '
    'If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
)
VERIFIER_PASS_TAG = "Final Decision: Yes"


def extract_last_boxed(text: str) -> str | None:
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    return matches[-1].group(1)


def extract_last_final_answer(text: str) -> str | None:
    candidate_patterns = [
        r"Final Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Final Answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"The answer is:\s*((?:[^<]|<[^<])*?)\n",
        r"Answer:\s*((?:[^<]|<[^<])*?)\n",
        r"Solution:\s*((?:[^<]|<[^<])*?)\n",
        r"The solution is:\s*((?:[^<]|<[^<])*?)\n",
    ]
    last_match: str | None = None
    last_position = -1
    for pattern in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if match.start() > last_position:
                last_position = match.start()
                last_match = match.group(1).strip()

    if last_match is None:
        return None
    for stop_word in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if last_match.endswith(stop_word):
            last_match = last_match[: -len(stop_word)].strip()
    return last_match


def extract_solution(solution_str: str) -> str | None:
    return extract_last_boxed(solution_str) or extract_last_final_answer(solution_str)


def _sleep_llm(llm: LLM) -> None:
    sleep = getattr(llm, "sleep", None)
    if callable(sleep):
        sleep(2)


class RewardModelWorker(Worker):
    """vLLM verifier worker compatible with verl RewardModel role."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        max_tokens = int(config.get("sampling_max_tokens", 2048))
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        gpu_memory_utilization = float(self.config.get("gpu_memory_utilization", 0.5))
        self.llm = LLM(model=self.config.model.path, gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = hf_tokenizer(
            self.config.model.path,
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )
        _sleep_llm(self.llm)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto) -> DataProto:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wake_up = getattr(self.llm, "wake_up", None)
        if callable(wake_up):
            wake_up()

        sequence_strs: list[str] = []
        ground_truths: list[str] = []
        questions: list[str] = []
        valid_response_lengths: list[int] = []

        for index in range(len(data)):
            data_item = data[index]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_lengths.append(valid_response_length)

            sequence = torch.cat((valid_prompt_ids, response_ids[:valid_response_length]))
            sequence_strs.append(self.tokenizer.decode(sequence[-1024:]))

            extra_info = data_item.non_tensor_batch["extra_info"]
            questions.append(str(extra_info["question"]))
            ground_truths.append(str(data_item.non_tensor_batch["reward_model"]["ground_truth"]))

        solutions = [extract_solution(sequence) for sequence in sequence_strs]
        messages = [
            VERIFIER_PROMPT_TEMPLATE.format(
                question=question,
                ground_truth=ground_truth,
                student_answer=solution or "No Answer",
            )
            for question, ground_truth, solution in zip(questions, ground_truths, solutions, strict=True)
        ]
        outputs = self.llm.generate(messages, self.sampling_params)
        verifications = [output.outputs[0].text.strip() for output in outputs]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for index, (ground_truth, solution, verification, valid_response_length) in enumerate(
            zip(ground_truths, solutions, verifications, valid_response_lengths, strict=True)
        ):
            score = 0.0
            if solution is None:
                score -= 0.5
                solution = "No Answer"
            if VERIFIER_PASS_TAG in verification:
                score += 1.0
                solution_tokens = self.tokenizer.encode(solution)
                ground_truth_tokens = self.tokenizer.encode(ground_truth)
                token_delta = min(abs(len(solution_tokens) - len(ground_truth_tokens)), 10)
                score -= token_delta * 0.05
            reward_tensor[index, valid_response_length - 1] = score

        batch = TensorDict({"rm_scores": reward_tensor}, batch_size=reward_tensor.shape[0])
        _sleep_llm(self.llm)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return DataProto(batch=batch)
