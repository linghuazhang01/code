"""Revision-3 answer markers must follow prose/boxed-line position rules."""

from __future__ import annotations

import pytest
from test_structure_positions import _piece_mask

MATH_ANSWERS = [1590, 4226, 13023, 16688, 19357, 21806, 73877]
CODE_ANSWERS = [1590, 4226, 9217, 11822, 13023, 16688, 19357, 21806]


@pytest.mark.parametrize(
    "domain,token_id",
    [("math", token_id) for token_id in MATH_ANSWERS]
    + [("code", token_id) for token_id in CODE_ANSWERS],
)
def test_each_answer_marker_requires_heading_or_closed_bold_label(
    domain: str, token_id: int
) -> None:
    # Code labels remain eligible even when the response has no fenced block.
    assert _piece_mask(
        [
            "the ",
            token_id,
            " appears in prose.\n## ",
            token_id,
            "\n**",
            token_id,
            ":** after ",
            token_id,
            " in prose.\n__",
            token_id,
            "__\n",
            token_id,
            " is plain.\n",
        ],
        domain,
    ) == [False, True, True, False, True, False]


@pytest.mark.parametrize("token_id", MATH_ANSWERS)
def test_math_boxed_line_only_and_literal_backslash_required(token_id: int) -> None:
    assert _piece_mask(
        [
            "the ",
            token_id,
            " is \\",
            79075,
            "{1}, so ",
            token_id,
            ".\n",
            token_id,
            " is prose on the next line.\n",
            token_id,
            " appears with ",
            79075,
            "{2} but no backslash.\n",
            token_id,
            " is before the boxed line.\n\\",
            79075,
            "{3}\n",
        ],
        "math",
    ) == [True, True, True, False, False, True, False, True]


@pytest.mark.parametrize("token_id", CODE_ANSWERS)
def test_code_answer_variables_comments_strings_and_docstrings_stay_unweighted(
    token_id: int,
) -> None:
    assert _piece_mask(
        [
            "## ",
            token_id,
            "\n```python\n",
            1350,
            "(",
            token_id,
            ")\n",
            token_id,
            " = ",
            1355,
            "()\n",
            750,
            " f(",
            token_id,
            "):\n    return ",
            token_id,
            "\n# **",
            token_id,
            '**\ntext = "**',
            token_id,
            "**\"\ndoc = '''\n## ",
            token_id,
            "\n'''\n```\n__",
            token_id,
            "__\n",
        ]
    ) == [True, True, False, False, True, True, False, False, False, False, False, True]


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_code_labels_inside_unclosed_block_are_excluded(fence: str) -> None:
    assert _piece_mask(
        ["## ", 16688, f"\n{fence}python\n## ", 16688, "\n**", 16688, "**\n"]
    ) == [True, False, False]


@pytest.mark.parametrize("token_id", CODE_ANSWERS)
def test_code_prose_does_not_gain_math_boxed_line_exception(token_id: int) -> None:
    assert _piece_mask(["the ", token_id, " is \\boxed{1}.\n"]) == [False]


def test_math_answer_suffixes_and_unclosed_bold_are_not_labels() -> None:
    assert _piece_mask(
        ["##", 1590, "e\n**", 4226, " without closing\n##", 16688, "s\n"],
        "math",
    ) == [False, False, False]


def test_boxed_command_boundary_and_crlf_line_endings() -> None:
    assert _piece_mask(
        [
            "the",
            4226,
            " is \\boxedness{1}.\r\n",
            "the",
            4226,
            " is \\boxed{2}.\r\n",
            "the",
            4226,
            " is on the next line.\r\n",
        ],
        "math",
    ) == [False, True, False]
