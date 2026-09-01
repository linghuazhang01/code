"""Paper-aligned prompt builders for code evaluation datasets."""

from __future__ import annotations

EVALPLUS_CODE_INSTRUCTION = (
    "Write Python code to solve the problem. Present the code in \n"
    "```python\n"
    "Your code\n"
    "```\n"
    "at the end.\n"
    "You need to think first then write the Python code."
)
EVALPLUS_PROMPT_TEMPLATE = "{task_prompt}\n\n\n" + EVALPLUS_CODE_INSTRUCTION

LCB_QWEN3_PREAMBLE = (
    "You will be given a question (problem specification) and will generate a correct "
    "Python program that matches the specification and passes all tests. You will NOT "
    "return anything except for the program.\n\n"
)
LCB_QWEN3_PROMPT_TEMPLATE = (
    f"{LCB_QWEN3_PREAMBLE}"
    "Question:\n{question_content}\n\n\n\n"
    f"{EVALPLUS_CODE_INSTRUCTION}"
)

LCB_AZR_SYSTEM_MESSAGE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: "
)

LCB_FORMATTING_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)

LCB_FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters "
    "as follows. Ensure that when the python program runs, it reads the inputs, runs "
    "the algorithm and writes output to STDOUT."
)


def build_evalplus_prompt(task_prompt: str) -> str:
    """Build the exact user content produced by the G-OPD EvalPlus path."""

    return EVALPLUS_PROMPT_TEMPLATE.format(task_prompt=task_prompt.strip())


def build_lcb_qwen3_non_thinking_prompt(question_content: str) -> str:
    """Build the exact G-OPD Qwen3NonThinking LiveCodeBench eval prompt."""

    return LCB_QWEN3_PROMPT_TEMPLATE.format(question_content=question_content)


def build_lcb_azr_prompt(question_content: str, starter_code: str = "") -> str:
    """Build the LiveCodeBench AZR prompt content kept for script-level reproduction."""

    prompt = (
        "\n# Task: You will be given a question (problem specification) and will generate "
        "a correct Python program that matches the specification and passes all tests. "
        "Your final answer should be wrapped in ```python``` tags.\n\n"
        f"Question: {question_content.strip()}\n\n"
    )
    if starter_code.strip():
        prompt += f"{LCB_FORMATTING_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{starter_code.strip()}\n```\n\n"
    else:
        prompt += f"{LCB_FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "Assistant: <think>"
    return f"{LCB_AZR_SYSTEM_MESSAGE}{prompt}"
