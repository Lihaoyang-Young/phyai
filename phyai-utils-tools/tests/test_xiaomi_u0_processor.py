"""CPU contracts for the fixed-grid U0 processor."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from phyai_utils_tools.models.xiaomi_u0 import U0TextProcessor, visual_codes


@pytest.fixture
def image_tokens() -> torch.Tensor:
    """Construct a complete image with distinct first and last codebook IDs."""
    codes = torch.arange(4096, dtype=torch.int64)
    codes[-1] = 131071
    tokens = [151852, 21, 19, 9, 21, 19, 151851]
    for row in range(64):
        tokens.extend((codes[row * 64 : (row + 1) * 64] + 151854).tolist())
        if row != 63:
            tokens.append(151846)
    return torch.tensor(tokens + [151853, 151747, 151850], dtype=torch.int64)


def test_visual_codes_preserve_row_order_and_codebook_edges(
    image_tokens: torch.Tensor,
) -> None:
    """Extract row-major codes, including both ends of the codebook."""
    codes = visual_codes(image_tokens)
    assert codes.shape == (64, 64)
    assert codes.dtype == torch.int64
    assert codes[0, 0].item() == 0
    assert codes[0, -1].item() == 63
    assert codes[1, 0].item() == 64
    assert codes[-1, -1].item() == 131071


@pytest.mark.parametrize(
    "index,value",
    [
        (0, 151853),
        (71, 151847),
        (7, 151853),
        (7, 282926),
        (4166, 151846),
        (4167, 151850),
        (4168, 151747),
    ],
)
def test_visual_codes_reject_malformed_sequence(
    image_tokens: torch.Tensor,
    index: int,
    value: int,
) -> None:
    """Reject corrupt headers, row separators, code IDs and termination."""
    image_tokens[index] = value
    with pytest.raises(ValueError):
        visual_codes(image_tokens)


def test_visual_codes_reject_incomplete_or_batched_input(
    image_tokens: torch.Tensor,
) -> None:
    """Reject incomplete streams and tensors outside the public contract."""
    for invalid in (image_tokens[:-1], image_tokens.unsqueeze(0), image_tokens.float()):
        with pytest.raises(ValueError):
            visual_codes(invalid)


@pytest.fixture
def checkpoint_processor() -> U0TextProcessor:
    """Enable real-vocabulary integration tests only with an explicit checkpoint."""
    value = os.environ.get("U0_TEST_CHECKPOINT")
    if not value:
        pytest.skip("Set U0_TEST_CHECKPOINT to run real-vocabulary tests.")
    return U0TextProcessor(Path(value))


def test_checkpoint_mapping(checkpoint_processor: U0TextProcessor) -> None:
    """Verify the released fixed-grid header encoding."""
    mapping = checkpoint_processor.header_mapping()
    assert mapping["64*64"] == [21, 19, 9, 21, 19]
    assert mapping["*"] == [9]
    assert all(mapping[str(index)] == [15 + index] for index in range(10))


@pytest.mark.parametrize(
    "prompt,length",
    [
        (
            "A clean product photograph of a glass teapot beside green tea leaves, soft daylight, neutral background.",
            40,
        ),
        (
            "A compact industrial inspection camera on a brushed steel workbench, precise studio lighting.",
            36,
        ),
        (
            "A white robotic gripper holding a translucent blue cube above a matte black calibration board.",
            37,
        ),
    ],
)
def test_checkpoint_prompt_contract(
    checkpoint_processor: U0TextProcessor,
    prompt: str,
    length: int,
) -> None:
    """Verify the previously accepted prompt lengths and task delimiters."""
    conditional, unconditional = checkpoint_processor.build_inputs(prompt)
    assert conditional.shape == (1, length)
    assert unconditional.shape == (1, 21)
    assert conditional.dtype == unconditional.dtype == torch.int64
    assert conditional[0, 0].item() == unconditional[0, 0].item() == 151849
    assert conditional[0, -1].item() == unconditional[0, -1].item() == 151746


@pytest.mark.parametrize("prompt", ["", "  ", "<|image start|>", "<|extra_204|>"])
def test_checkpoint_reject_reserved_or_empty_prompt(
    checkpoint_processor: U0TextProcessor,
    prompt: str,
) -> None:
    """User text cannot inject task-control tokens into the template."""
    with pytest.raises(ValueError):
        checkpoint_processor.build_inputs(prompt)
