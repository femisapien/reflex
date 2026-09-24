"""Flat, one-token choice labels for wide SGLang requests."""

from __future__ import annotations

import itertools
import random
import string
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from reflex.prompt import (
    Branch,
    PromptFormat,
    _compact_body,
    _options_block,
    build_branches,
    distinct_orders,
    render_text,
)
from reflex.schema import ChoiceQuestion, Text


@lru_cache(maxsize=16)
def _wide_alphabet(tokenizer: Any) -> tuple[str, ...]:
    return tuple(
        candidate
        for pair in itertools.product(string.ascii_uppercase, repeat=2)
        if len(tokenizer.encode(candidate := "".join(pair), add_special_tokens=False)) == 1
    )


def choice_option_capacity(tokenizer: Any) -> int:
    """Largest choice supported by this tokenizer's exact one-token readout."""
    # The original A-Z prompts remain the readout for narrow choices. Validate
    # them as well, so the reported capacity really is usable from 2 upward.
    single_token_labels(tokenizer, 26)
    return max(26, len(_wide_alphabet(tokenizer)))


def single_token_labels(tokenizer: Any, count: int) -> list[str]:
    """Choose distinct one-token labels, preserving A-Z for existing requests.

    Wide requests use only two-letter labels so every option has the same label
    length. The tokenizer determines which labels are actually a single output
    token. Numbers do not work for Qwen: its pre-tokenizer isolates each digit.
    """
    if count < 2:
        raise ValueError("a choice needs at least two options")
    if count <= 26:
        labels = list(string.ascii_uppercase[:count])
    else:
        alphabet = _wide_alphabet(tokenizer)
        if len(alphabet) < count:
            raise ValueError(
                f"tokenizer has only {len(alphabet)} suitable one-token labels; "
                f"{count} options require another readout"
            )
        labels = list(alphabet[:count])

    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != count:
        raise ValueError("labels must map one-to-one to single token ids")
    return labels


def build_choice_branches(
    qid: str,
    instructions: Text,
    criteria: Mapping[str, Text | None],
    fmt: PromptFormat,
    tokenizer: Any,
    permutations: int = 2,
    seed: int = 0,
) -> list[Branch]:
    """Render all options with a flat selected-token readout per permutation.

    For <=26 options, call the unmodified Reflex builder for byte-identical
    prompts. For wider choices, use tokenizer-verified labels. Each branch sees
    *all* the options and scores one token per option; nothing is truncated or
    partitioned into uncalibrated buckets.
    """
    keys = list(criteria)
    labels = single_token_labels(tokenizer, len(keys))
    if len(keys) <= 26:
        q = ChoiceQuestion(type="choice", instructions=instructions, criteria=dict(criteria))
        return build_branches(qid, q, fmt, permutations=permutations, seed=seed)

    rng = random.Random(f"{seed}:{qid}")
    branches: list[Branch] = []
    for order in distinct_orders(keys, permutations, rng):
        labelled = []
        for label, key in zip(labels, order, strict=True):
            description = render_text(criteria[key]) if criteria[key] is not None else ""
            option_text = f"{key}: {description}" if description else key
            labelled.append((label, option_text))
        if fmt.style == "compact":
            body = _compact_body(instructions, labelled)
        else:
            body = _options_block(
                instructions,
                labelled,
                "Respond with only the two-letter label of the best option.",
                fmt,
            )
        branches.append(Branch(qid, "choice", fmt.branch(body), labels.copy(), order))
    return branches
