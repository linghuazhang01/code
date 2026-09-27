"""Conservative Markdown/Python spans for versioned token position gates."""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass


@dataclass(frozen=True)
class FencedCodeBlock:
    opening: tuple[int, int]
    closing: tuple[int, int] | None
    end: int


def fenced_code_blocks(text: str) -> list[FencedCodeBlock]:
    """Include open-ended blocks; closing fences must match the opening kind."""
    blocks: list[FencedCodeBlock] = []
    opening: tuple[int, int] | None = None
    fence = ""
    for line in re.finditer(r"(?m)^.*(?:\n|$)", text):
        if line.start() == len(text):
            continue
        stripped = line.group().strip()
        if opening is None:
            match = re.fullmatch(r"(`{3,}|~{3,})([^\r\n]*)", stripped)
            if match and not (match[1][0] == "`" and "`" in match[2]):
                opening, fence = line.span(), match[1]
        elif re.fullmatch(re.escape(fence[0]) + "{" + str(len(fence)) + ",}", stripped):
            blocks.append(FencedCodeBlock(opening, line.span(), line.end()))
            opening = None
    if opening is not None:
        blocks.append(FencedCodeBlock(opening, None, len(text)))
    return blocks


def _eligible_statement(tokens: list[tokenize.TokenInfo]) -> bool:
    """Accept I/O calls/aliases, function signatures, and explicit entry lines."""
    meaningful = [
        token
        for token in tokens
        if token.type
        not in {
            tokenize.COMMENT,
            tokenize.STRING,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENDMARKER,
        }
    ]
    values = [token.string for token in meaningful]
    if not values:
        return False
    if values[0] == "def" or values[:2] == ["async", "def"]:
        return True
    if values[:2] == ["if", "__name__"] and any(
        token.type == tokenize.STRING and token.string.strip("'\"") == "__main__"
        for token in tokens
    ):
        return True
    joined = "".join(values)
    if re.fullmatch(r"(?:solve|main)\(\)", joined):
        return True
    # Join is an output-formatting operation; split/int/map by themselves are
    # ordinary computation and do not make a line eligible.
    for index, value in enumerate(values):
        if values[index : index + 3] in (["sys", ".", "stdin"], ["sys", ".", "stdout"]):
            return True
        if index + 1 == len(values) or values[index + 1] != "(":
            continue
        attribute = index > 0 and values[index - 1] == "."
        if value in {"read", "readline"} or (
            value in {"input", "print"} and not attribute
        ):
            return True
        if value == "join" and attribute:
            return True
    return False


def code_statement_spans(
    text: str,
    start: int,
    end: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return eligible lexical spans and excluded comments/string spans.

    Logical statements cover multiline calls/signatures. Incomplete Python
    tokenization fails closed for body tokens; fences and EOS remain independent.
    """
    body = text[start:end]
    line_starts = [start]
    line_starts.extend(start + match.end() for match in re.finditer("\n", body))

    def span(token: tokenize.TokenInfo) -> tuple[int, int]:
        return (
            line_starts[token.start[0] - 1] + token.start[1],
            line_starts[token.end[0] - 1] + token.end[1],
        )

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(body).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [], [(start, end)]
    if any(
        token.type == tokenize.ERRORTOKEN and token.string.strip() for token in tokens
    ):
        return [], [(start, end)]
    excluded = [
        span(token)
        for token in tokens
        if token.type in {tokenize.COMMENT, tokenize.STRING}
    ]
    eligible: list[tuple[int, int]] = []
    statement: list[tokenize.TokenInfo] = []
    for token in tokens:
        statement.append(token)
        if token.type not in {tokenize.NEWLINE, tokenize.ENDMARKER}:
            continue
        if _eligible_statement(statement):
            eligible.extend(
                span(item)
                for item in statement
                if item.type in {tokenize.NAME, tokenize.NUMBER, tokenize.OP}
            )
        statement = []
    return eligible, excluded
