"""Position rules for the fixed Token V4/V5 Structure category."""

from __future__ import annotations

import pytest
import torch

from mopd_verl.domain_gradient.structure_positions import (
    control_position_mask,
    structure_position_mask,
)
from mopd_verl.domain_gradient.token_taxonomy_registry import token_taxonomy


class _Tokenizer:
    def __init__(self, surfaces: dict[int, str]):
        self.surfaces = surfaces
        self.last_ids: list[int] = []

    def decode(self, token_ids: list[int], **_: object) -> str:
        self.last_ids = list(token_ids)
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def __call__(self, text: str, **_: object) -> dict[str, object]:
        raise AssertionError("Structure parsing must not retokenize generated IDs.")


def test_math_labels_require_heading_or_bold_but_boxed_and_eos_are_anywhere() -> None:
    surfaces = {
        19357: "Final",
        21806: " Answer",
        79075: "boxed",
        151645: "<|im_end|>",
        9000: " appears in prose.\n## ",
        9001: "\nplain ",
        9002: "\n",
    }
    ids = torch.tensor([[19357, 9000, 19357, 21806, 9001, 79075, 9002, 151645]])
    mask = structure_position_mask(
        ids,
        torch.ones_like(ids),
        ["math"],
        _Tokenizer(surfaces),
        {"math": [1590, 13023, 19357, 21806, 79075, 151645]},
    )
    assert mask.tolist() == [[False, False, True, True, False, True, False, True]]


def test_noncanonical_prefix_does_not_disable_later_structure_positions() -> None:
    # Qwen can decode [13023, 68] as " Finale" but re-encode it with different
    # IDs. Original-ID offsets must still allow a later heading label to apply.
    surfaces = {
        13023: " Final",
        68: "e",
        9000: " in prose.\n##",
        9001: "\n",
        151645: "<|im_end|>",
    }
    ids = torch.tensor([[13023, 68, 9000, 13023, 9001, 151645]])
    mask = structure_position_mask(
        ids,
        torch.ones_like(ids),
        ["math"],
        _Tokenizer(surfaces),
        {"math": [13023, 151645]},
    )
    assert mask.tolist() == [[False, False, False, True, False, True]]


def test_code_uses_only_final_closed_block_and_excludes_comment_lines() -> None:
    surfaces = {
        73594: "```",
        12669: "python",
        1350: "print",
        750: "def",
        151645: "<|im_end|>",
        9000: "\n",
        9001: "(1)\n",
        9002: "text\n",
        9003: "# ",
        9004: " f():\n",
    }
    ids = torch.tensor(
        [
            [
                73594,
                12669,
                9000,
                1350,
                9001,
                73594,
                9000,
                9002,
                73594,
                12669,
                9000,
                9003,
                1350,
                9001,
                750,
                9004,
                73594,
                9000,
                151645,
            ]
        ]
    )
    structure_ids = [73594, 12669, 1350, 750, 151645]
    mask = structure_position_mask(
        ids,
        torch.ones_like(ids),
        ["code"],
        _Tokenizer(surfaces),
        {"code": structure_ids},
    )
    expected = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert mask.tolist() == [expected]


def _piece_mask(pieces: list[int | str], domain: str = "code") -> list[bool]:
    surfaces = {
        396: "int",
        526: " int",
        750: "def",
        1173: " print",
        1350: "print",
        1355: "input",
        1590: " final",
        4226: " answer",
        5987: "join",
        6960: "split",
        7791: "sys",
        9217: "answer",
        11822: "final",
        13023: " Final",
        16688: " conclusion",
        19357: "Final",
        21806: " Answer",
        31800: ".readline",
        43184: ".stdin",
        59419: "solve",
        73594: "```",
        73877: " Conclusion",
        79075: "boxed",
        151645: "<|im_end|>",
    }
    ids = []
    for index, piece in enumerate(pieces):
        if isinstance(piece, str):
            surfaces[200000 + index] = piece
            ids.append(200000 + index)
        else:
            ids.append(piece)
    tensor = torch.tensor([ids])
    result = structure_position_mask(
        tensor,
        torch.ones_like(tensor),
        [domain],
        _Tokenizer(surfaces),
        {domain: token_taxonomy("v5")[domain].structure},
    )[0].tolist()
    return [
        accepted for piece, accepted in zip(pieces, result) if isinstance(piece, int)
    ]


def test_code_only_io_lines_excludes_calculation_comments_and_string_tokens() -> None:
    assert _piece_mask(
        [
            "```python\nx = ",
            396,
            "(value)\ns = value.",
            6960,
            "()\n",
            "n = ",
            396,
            "(",
            1355,
            "()).",
            6960,
            "() # ",
            1355,
            "()\n",
            1350,
            '("',
            526,
            '")\n',
            'doc = """',
            750,
            " fake():\n",
            1355,
            '()"""\n',
            "```\n",
        ]
    ) == [False, False, True, True, True, False, True, False, False, False]


def test_code_multiline_io_signature_alias_and_entry_lines() -> None:
    assert _piece_mask(
        [
            "```python\n",
            1355,
            " = ",
            7791,
            43184,
            31800,
            "\n",
            750,
            " f(\n    value:",
            526,
            "\n):\n    x = ",
            396,
            "(value)\n",
            1350,
            "(\n    ",
            396,
            "(",
            1355,
            "())\n)\n",
            59419,
            "()\n```\n",
        ]
    ) == [True, True, True, True, True, True, False, True, True, True, True]


def test_output_join_excludes_literal_and_fence_words_inside_body() -> None:
    assert _piece_mask(
        [
            "```python\n",
            1350,
            "(' '.",
            5987,
            "(result))\n",
            'text = "',
            73594,
            '"\n```\n',
        ]
    ) == [True, True, False]


def test_math_lowercase_heading_and_bold_label_end_before_following_prose() -> None:
    assert _piece_mask(
        [
            "The",
            1590,
            " result.\n##",
            1590,
            21806,
            "\n**",
            1590,
            "** then",
            1590,
            " in prose.\n__",
            19357,
            21806,
            ":__ result",
        ],
        "math",
    ) == [False, True, True, True, False, True, True]


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_control_excludes_all_fenced_blocks_including_unclosed_tail(fence: str) -> None:
    surfaces = {333: "if", 9000: f"\n{fence}python\n", 9001: f"\n{fence}\n"}
    ids = torch.tensor([[333, 9000, 333, 9001, 333, 9000, 333]])
    mask = control_position_mask(
        ids, torch.ones_like(ids), ["code"], _Tokenizer(surfaces)
    )
    assert mask[0, [0, 2, 4, 6]].tolist() == [True, False, True, False]


def test_final_closed_block_survives_trailing_unclosed_block() -> None:
    assert _piece_mask(
        ["```python\n", 1350, "(1)\n```\n```python\n", 1350, "(2)\n"]
    ) == [True, False]


def test_unterminated_string_in_closed_block_fails_body_closed() -> None:
    assert _piece_mask(["```python\n", 1350, "(1)\ntext = '\n", 1355, "()\n```\n"]) == [
        False,
        False,
    ]
