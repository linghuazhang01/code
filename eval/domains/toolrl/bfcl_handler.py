"""BFCL RLLA handler adapted from the ToolRL official repository."""

from __future__ import annotations

import json
from typing import Any

try:
    from bfcl.model_handler.local_inference.base_oss_handler import OSSHandler
    from overrides import override
except ImportError:
    OSSHandler = object  # type: ignore[assignment]

    def override(func):  # type: ignore[no-untyped-def]
        return func


JSON_STRING = (
    '{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}\n'
    '{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}'
)

SYSTEM_TEMPLATE = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls
to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{json_string}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning.
2. Each tool call should be a JSON object with a "name" field and a "parameters" dictionary.
3. Refer to previous dialogue records and tool feedback noted as `<obs>` when available.
"""


def _format_tool(tool: Any, count: int = 1) -> str:
    if isinstance(tool, dict):
        params = dict(tool.get("parameters", {}).get("properties", {}))
        return (
            f"{count}. Name: {tool['name']}\n"
            f"Description: {tool['description']}\n"
            f"Parameters: {json.dumps(params, ensure_ascii=False)}"
        )
    if isinstance(tool, list):
        return "\n".join(_format_tool(item, index + 1) for index, item in enumerate(tool))
    return str(tool)


class RLLAHandler(OSSHandler):
    """BFCL handler using ToolRL/RLLA prompt and decoding format."""

    def __init__(self, model_name: str, temperature: float) -> None:
        if OSSHandler is object:
            raise RuntimeError("BFCL is not installed. Install the BFCL harness before using RLLAHandler.")
        super().__init__(model_name, temperature)

    @override
    def _format_prompt(
        self,
        messages: list[dict[str, Any]],
        function: Any,
        turn_type: str = "single_turn",
    ) -> str:
        tools = _format_tool(function)
        system_prompt = SYSTEM_TEMPLATE.format(tools=tools, json_string=JSON_STRING)
        user_prompt = "**Dialogue Records History**\n"
        for message_index, message in enumerate(messages):
            role = message.get("role")
            if role == "system":
                continue
            if role == "user":
                suffix = (
                    "Use the one or more necessary tool calls to complete the task. You could perform tool calls "
                    "for multiple rounds so you can try and error."
                    if turn_type == "multi_turn"
                    else "If no tools apply or required parameters are missing, directly inform me without a tool call."
                )
                user_prompt += f"<user> {str(message['content']).strip()}\n{suffix} </user>\n"
            elif role == "tool":
                tool_name = str(message.get("name", "")).strip()
                tool_result = str(message.get("content", "")).strip()
                user_prompt += (
                    f"<obs> You have made the tool call {tool_name}. "
                    f"Execution returns: {tool_result} </obs>\n"
                )
                if message_index == len(messages) - 1:
                    user_prompt += (
                        "<user> If the task is complete, respond directly. "
                        "Otherwise retry with tools. </user>\n"
                    )
            elif role == "assistant":
                user_prompt += f"\n{str(message.get('content', '')).strip()}\n"
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt.strip()}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @override
    def decode_ast(self, result: str, language: str = "Python") -> list[dict[str, Any]]:
        if "<tool_call>" not in result:
            return []
        decoded_output: list[dict[str, Any]] = []
        calls = result.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip().splitlines()
        for call in calls:
            try:
                tool_call = json.loads(call)
                name = tool_call["name"]
                if name.strip().lower() == "none":
                    continue
                decoded_output.append({name: tool_call["parameters"]})
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return decoded_output

    @override
    def decode_execute(self, result: str) -> list[str]:
        if "<tool_call>" not in result:
            return []
        tool_calls = []
        calls = result.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip().splitlines()
        for call in calls:
            try:
                tool_call = json.loads(call)
                if str(tool_call.get("name", "")).strip().lower() != "none":
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        return self.xlam_json_to_python_tool_calls(tool_calls)

    @staticmethod
    def xlam_json_to_python_tool_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
        output = []
        for tool_call in tool_calls:
            name = str(tool_call.get("name", ""))
            arguments = dict(tool_call.get("parameters") or {})
            args_str = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
            output.append(f"{name}({args_str})")
        return output
