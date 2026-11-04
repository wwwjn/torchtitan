# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import json

import numpy as np
import pytest

from torchtitan.rl.examples.slake import (
    RewardSlakeAnswer,
    SlakeDataset,
    SlakeEnv,
    SlakeSample,
)
from torchtitan.rl.examples.slake.rubric import is_correct, normalize_answer
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.types import RolloutTurnID


def _row(qid: int, **overrides) -> dict:
    row = {
        "qid": qid,
        "img_name": f"xmlab{qid}/source.jpg",
        "question": f"Question {qid}?",
        "answer": "Yes",
        "q_lang": "en",
        "answer_type": "CLOSED",
        "content_type": "Organ",
    }
    row.update(overrides)
    return row


@pytest.fixture
def data_dir(tmp_path):
    rows = [
        _row(0),
        _row(1, answer="Liver", content_type="Size"),
        _row(2, answer_type="OPEN", answer="CT"),
        _row(3, content_type="KG"),
        _row(4, q_lang="zh"),
    ]
    (tmp_path / "train.json").write_text(json.dumps(rows))
    return tmp_path


def test_dataset_keeps_english_closed_image_questions(data_dir) -> None:
    dataset = SlakeDataset(SlakeDataset.Config(data_dir=str(data_dir)))
    assert len(dataset) == 2
    samples = [next(dataset) for _ in range(4)]
    # Two reshuffled epochs, each a permutation of the kept questions.
    assert {s.qid for s in samples[:2]} == {s.qid for s in samples[2:]} == {0, 1}
    assert samples[0].image_path == str(
        data_dir / "imgs" / f"xmlab{samples[0].qid}/source.jpg"
    )

    open_dataset = SlakeDataset(
        SlakeDataset.Config(data_dir=str(data_dir), closed_only=False)
    )
    assert len(open_dataset) == 3


def test_dataset_state_dict_resumes_the_stream(data_dir) -> None:
    config = SlakeDataset.Config(data_dir=str(data_dir), seed=7)
    dataset = SlakeDataset(config)
    next(dataset)
    state = dataset.state_dict()
    expected = [next(dataset).qid for _ in range(5)]

    resumed = SlakeDataset(config)
    resumed.load_state_dict(state)
    assert [next(resumed).qid for _ in range(5)] == expected


@pytest.mark.parametrize("include_image", [True, False])
def test_env_asks_one_question(include_image: bool) -> None:
    sample = SlakeSample(
        qid=0,
        image_path="/data/imgs/a.jpg",
        question="Is the lung healthy?",
        answer="Yes",
        content_type="Abnormality",
    )
    env = SlakeEnv(SlakeEnv.Config(include_image=include_image), env_input=sample)
    init = asyncio.run(env.init())
    (message,) = init.init_prompt_messages
    if include_image:
        image_part, text_part = message["content"]
        assert image_part == {"type": "image", "path": "/data/imgs/a.jpg"}
        assert text_part["text"].startswith("Is the lung healthy?\n")
    else:
        assert message["content"].startswith("Is the lung healthy?\n")
    step = asyncio.run(
        env.step({"role": "assistant", "content": "<answer>Yes</answer>"})
    )
    assert step.done


@pytest.mark.parametrize(
    ("response", "expected", "correct"),
    [
        ("<answer>Yes</answer>", "Yes", True),
        ("<answer> yes, it does. </answer>", "Yes", True),
        ("<answer>No</answer>", "Yes", False),
        ("<answer>The axial plane</answer>", "Transverse Plane", True),
        ("<answer>X ray</answer>", "X-Ray", True),
        ("<answer>left lung</answer>", "Lung", False),
        ("<answer>No</answer> wait <answer>Liver</answer>", "Liver", True),
        ("Yes", "Yes", False),
    ],
)
def test_is_correct(response: str, expected: str, correct: bool) -> None:
    assert is_correct(response, expected) is correct


def test_normalize_answer() -> None:
    assert normalize_answer(" The Axial plane. ") == "transverse plane"
    assert normalize_answer("T2-weighted") == "t2"
    assert normalize_answer("Small Bowel") == "small bowel"


def test_reward_scores_the_last_turn() -> None:
    sample = SlakeSample(
        qid=0, image_path="a.jpg", question="q", answer="Liver", content_type="Size"
    )
    turn = RolloutTurn(
        rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
        prompt_prefix_len=0,
        prompt_delta_token_ids=np.array([1], dtype=np.int32),
        completion_token_ids=np.array([2], dtype=np.int32),
        completion_logprobs=np.array([0.0], dtype=np.float32),
        completion_message={"role": "assistant", "content": "<answer>liver</answer>"},
    )
    rollout = Rollout(
        group_id=0, rollout_id=0, status=RolloutStatus.COMPLETED, turns=[turn]
    )
    reward = RewardSlakeAnswer(RewardSlakeAnswer.Config())
    assert asyncio.run(reward(rollout, sample)) == 1.0


def test_image_recipe_needs_images_on_generator_and_batcher() -> None:
    from torchtitan_recipes.rl.slake import rl_grpo_qwen3_5_0_8b_slake

    config = rl_grpo_qwen3_5_0_8b_slake()
    async_loop = config.async_loop
    batcher = dataclasses.replace(async_loop.batcher, images=None)

    with pytest.raises(ValueError, match="or neither"):
        dataclasses.replace(
            config, async_loop=dataclasses.replace(async_loop, batcher=batcher)
        )
