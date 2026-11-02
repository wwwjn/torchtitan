# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.rl.examples.slake.data import SlakeSample
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)

ANSWER_FORMAT = "Answer with a single word or a short phrase, inside <answer></answer>."


class SlakeEnv(MessageEnv):
    """Asks one SLAKE question about its image; the rollout ends after one answer.

    The user message holds the image followed by the question. With
    ``include_image=False`` the image is left out, which gives the text-only
    baseline (how much the model answers from priors alone).

    Example (with the image):

        init:  [image] "Does the picture contain liver?\\nAnswer with ..."
        step:  end the rollout
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        include_image: bool = True

    def __init__(self, config: Config, *, env_input: SlakeSample) -> None:
        self._include_image = config.include_image
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        """Ask the question, with the image first when enabled."""
        text = f"{self._env_input.question}\n{ANSWER_FORMAT}"
        content: str | list[dict] = text
        if self._include_image:
            content = [
                {"type": "image", "path": self._env_input.image_path},
                {"type": "text", "text": text},
            ]
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": content}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        """End the rollout: SLAKE questions are single-turn."""
        del completion_message
        return MessageEnvStepOutput(done=True)
