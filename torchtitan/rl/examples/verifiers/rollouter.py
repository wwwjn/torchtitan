# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A TitanRL rollout worker that runs Verifiers envs."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.env import RunSlot

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig, RenderersConfigAdapter

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.examples.verifiers.data import VerifiersTaskSample
from torchtitan.rl.examples.verifiers.generation_server import (
    GenerationServer,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.worker_setup import register_local_taskset_alias
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker
from torchtitan.rl.rollout.rubric import RewardFn
from torchtitan.rl.rollout.types import (
    GenerateFn,
    Prompt,
    ReleaseSessionFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
    split_prompt,
)

if TYPE_CHECKING:
    from torchtitan.rl.generator.types import SamplingConfig


logger = logging.getLogger(__name__)

VERIFIERS_REWARD_KEY = "verifiers_reward"


class RewardFromVerifiers(RewardFn):
    """Pass through the reward that Verifiers computed during the rollout."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        del env_input
        for turn in reversed(rollout.turns):
            if VERIFIERS_REWARD_KEY in turn.env_rewards:
                return float(turn.env_rewards[VERIFIERS_REWARD_KEY])
        return 0.0


class VerifiersRolloutWorker(BaseRolloutWorker):
    """Runs Verifiers envs in a TitanRL rollout worker.

        TitanRL provides     the tasks (`VerifiersTaskDataset`), the model, rewards, advantages
        Verifiers provides   the env, the agent loop, the sandbox, the grade

    The model reaches Verifiers through this worker's `GenerationServer`, which calls `generate_fn`.

    One group = one task, run `group_size` times with `env.run_slot`. Each episode's trainable
    trace becomes one TitanRL `Rollout`. How the pieces connect: "Integration flow" in this
    folder's README.md.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseRolloutWorker.Config):
        # TODO: enable multi-env training: load the env from each sample's config, so one job can mix
        # tasksets. Needs the dataset to carry each sample's env config. Until then, nothing checks that
        # this taskset matches the training or validation dataset's, though the worker builds both
        # datasets' tasks with it.
        environment: vf.EnvConfig
        """The Verifiers env each worker loads: its taskset, agent, harness and runtime. Give it
        the same taskset object as the training dataset. The validation dataset's taskset must
        have the same task class and task config: its tasks are built with this one."""

        generation_server: GenerationServer.Config
        """Local HTTP bridge from Verifiers to TitanRL generation; each rollout worker runs one."""

        renderer_multiplex: int = 256
        """Maximum concurrent rollouts sharing one Verifiers renderer instance.

        TODO: evaluate sharing this renderer pool with TitanRL's native rollout
        path instead of keeping it specific to the Verifiers client.
        """

        def __post_init__(self) -> None:
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._config = config
        self._generation_server: GenerationServer = config.generation_server.build()
        # A taskset under a local alias id can be imported only after this process registers it.
        taskset_module = _local_taskset_module(config.environment.taskset)
        if taskset_module is not None:
            register_local_taskset_alias(taskset_module)
        self._verifiers_env = vf.load_environment(
            vf.resolve_env_config(config.environment)
        )
        # Holds `env.serving()` open from `setup_async` to `close`. Those are two calls, so a `with`
        # block can't span them. Step 3 of `setup_async` says what it serves.
        self._serving = contextlib.AsyncExitStack()
        # Rollouts still running, so `close` can cancel them.
        self._running_rollouts: set[asyncio.Task[Rollout]] = set()
        self._train_client_config: vf.TrainClientConfig

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Start the local servers this worker's episodes use:

        1. `no_proxy` lists 127.0.0.1, so the local calls below skip the host's HTTP proxy.
        2. Our generation server: Verifiers' model calls -> `generate_fn`.
        3. `env.serving()`: Verifiers' interception server (the agent's model calls go through it)
           and its shared tools. It must stay open until `close`, so an `AsyncExitStack` holds it
           across the two calls.

        Verifiers builds its renderer from `renderer_config` and the tokenizer files in
        `hf_assets_path`, so `tokenizer_config` is unused.
        """
        del tokenizer_config
        if vision_processor_config is not None:
            # TODO: pass the vision processor to the renderer this worker's train client builds.
            raise NotImplementedError(
                "Verifiers environments do not support image prompts yet; unset "
                "Controller.Config.vision_processor."
            )
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError(
                "Verifiers' train client builds the renderer from a `renderers` config, the same "
                "config TitanRL uses: configure the renderer with from_renderers(...)"
            )
        # Verifiers' HTTP client sends even a 127.0.0.1 call through `http_proxy`, unless
        # `no_proxy` names 127.0.0.1. Add the loopback names to both spellings, keeping what
        # either already holds, e.g. ".corp.internal" -> ".corp.internal,127.0.0.1,localhost".
        no_proxy_entries = (
            os.environ.get("no_proxy"),
            os.environ.get("NO_PROXY"),
            "127.0.0.1",
            "localhost",
        )
        no_proxy = ",".join(dict.fromkeys(entry for entry in no_proxy_entries if entry))
        os.environ["no_proxy"] = os.environ["NO_PROXY"] = no_proxy
        # Verifiers' train client runs in this process, so it calls this server on the loopback
        # address.
        await self._generation_server.start()
        await self._serving.enter_async_context(self._verifiers_env.serving())
        self._train_client_config = vf.TrainClientConfig(
            base_url=self._generation_server.base_url,
            # The local endpoint does not authenticate. An empty environment
            # variable name makes Verifiers send its required "EMPTY" value.
            api_key_var="",
            renderer=renderer_config.get_renderers_config(),
            multiplex=self._config.renderer_multiplex,
            renderer_model_name=hf_assets_path,
        )

    async def close(self) -> None:
        """Cancel the rollouts still running, then stop serving the env and the generation server.
        Each cancelled `run_slot` stops its own sandbox, e.g. removes its Docker container."""
        # TODO: a killed worker never runs `close`, so its running sandboxes (e.g. Docker
        # containers) stay up. Removing them needs a cleanup that runs outside the worker.
        running_rollouts = list(self._running_rollouts)
        for rollout_task in running_rollouts:
            rollout_task.cancel()
        await asyncio.gather(*running_rollouts, return_exceptions=True)
        await self._serving.aclose()
        await self._generation_server.close()

    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Run one task `group_size` times through Verifiers, then score the group."""
        # The generation server holds one `generate_fn`, and each group sets it again. Safe: every
        # group's `generate_fn` routes the same way. Keep per-group state out of it.
        self._generation_server.set_generate_fn(generate_fn)

        # ======== Build the task ========
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError(
                f"Verifiers rollouts need a VerifiersTaskSample, got {type(sample).__name__}"
            )
        # The sample carries only the task's data; the env's taskset gives its class and config.
        task_type = self._verifiers_env.taskset.task_type()
        task = task_type(
            data=task_type.data_type().model_validate(sample.verifiers_task_data),
            config=self._verifiers_env.config.taskset.task,
        )

        # ======== Run the episodes ========
        # Verifiers plans one `RunSlot` per episode; `run_slot` runs it with the env's retries.
        slots = self._verifiers_env.slots(task, n=group_size)
        rollout_tasks = [
            asyncio.create_task(
                self._run_single_rollout(
                    release_session_fn=release_session_fn,
                    slot=slot,
                    sampling=(
                        sampling
                        if sampling.seed is None
                        else replace(sampling, seed=sampling.seed + rollout_id)
                    ),
                    group_id=group_id,
                    rollout_id=rollout_id,
                )
            )
            for rollout_id, slot in enumerate(slots)
        ]
        for rollout_task in rollout_tasks:
            self._running_rollouts.add(rollout_task)
            rollout_task.add_done_callback(self._running_rollouts.discard)
        # Wait for every rollout, even after one fails (why: `BaseRolloutWorker.run_group`).
        rollouts_or_errors = await asyncio.gather(
            *rollout_tasks, return_exceptions=True
        )

        # ======== Score the group ========
        group = await self.make_scored_group(
            rollouts_or_errors=rollouts_or_errors, sample=sample, group_id=group_id
        )
        # Count the errors Verifiers recorded on the group's rollouts, by type: verifiers/errors/<type>.
        # - A retried attempt's errors count too: Verifiers keeps them on the episode or the trace.
        # - A rollout that raised has only logs["error"], so it adds nothing here.
        # - Training only: validation doesn't log `group.metrics` yet (TODO in the controller).
        error_counts = Counter(
            error["type"]
            for rollout in group.rollouts
            for error in rollout.logs.get("errors", [])
        )
        group.metrics.extend(
            m.Metric(f"verifiers/errors/{error_type}", m.Sum(count))
            for error_type, count in sorted(error_counts.items())
        )
        return group

    async def _run_single_rollout(
        self,
        *,
        release_session_fn: ReleaseSessionFn,
        slot: RunSlot,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Run one Verifiers episode, release its sessions, and convert its trace to a rollout."""
        # ======== Run the episode and release its sessions ========
        model_context = vf.ModelContext(
            client=self._train_client_config,
            model=self._generation_server.model_id,
            sampling=vf.SamplingConfig(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=sampling.seed,
            ),
        )
        # Our generation server finds each model call's group by its session id: the X-Session-ID
        # header, which is the trace id. A retried attempt creates another trace, e.g.
        # session_ids == ["9f3c", "a71b"].
        session_ids: list[str] = []

        def start_session(trace: vf.Trace) -> None:
            # Verifiers calls this before the trace's first model call.
            session_ids.append(trace.id)
            self._generation_server.start_session(
                session_id=trace.id, group_id=group_id
            )

        try:
            verifiers_episode = await self._verifiers_env.run_slot(
                slot=slot, ctx=model_context, on_trace=start_session
            )
        finally:
            # TODO: release a session only after its model calls finish. Today this can happen:
            #   1. Verifiers gives up on a model call (e.g. at its deadline), but our generation
            #      server is still running it.
            #   2. The release below tells the router the session is done.
            #   3. The abandoned call then reaches the router, which sends the session to a
            #      generator again.
            #   4. That generator keeps the session's KV cache until the group's release.
            #   Fix: the generation server counts each session's running calls, and the release
            #   waits for the last one.
            generation_metadata_by_session = {
                session_id: self._generation_server.end_session(session_id)
                for session_id in session_ids
            }
            await asyncio.gather(
                *(
                    release_session_fn(routing_session_id=session_id)
                    for session_id in session_ids
                )
            )

        # ======== Pick the policy's trace ========
        # One trace per agent, e.g. the policy and a simulated user. Exactly one is trainable, the
        # policy's: keep it. A retried attempt's traces are gone; Verifiers keeps only the last
        # attempt.
        trainable_traces = [
            trace for trace in verifiers_episode.traces if trace.agent.trainable
        ]
        if len(trainable_traces) != 1:
            raise ValueError(
                "Verifiers expects one trainable trace per rollout; got "
                f"{len(trainable_traces)}"
            )
        (trace,) = trainable_traces

        # ======== Convert it to turns and a status ========
        # Verifiers' graph trace becomes TitanRL's linear rollout turns.
        turns = self.trace_to_rollout_turns(
            trace=trace,
            generation_metadata=generation_metadata_by_session[trace.id],
            group_id=group_id,
            rollout_id=rollout_id,
        )
        status = self.rollout_status(verifiers_episode=verifiers_episode, trace=trace)
        if not turns:
            # No turn: the policy took no action, so there is nothing to train.
            # If it were scored, the builder would also skip its whole group.
            if status.is_trainable():
                status = RolloutStatus.ERROR_INFRA
            # Verifiers records an overlong first prompt as a failed model call; vLLM, the
            # renderer and the generation server all word it "maximum context length".
            # TODO: drop from the dataset tasks whose first prompt doesn't fit, and detect the overflow
            # by the call's status 400 instead of its error text. Dropping needs the prompt before
            # the rollout, but the harness builds it in the worker.
            first_prompt_too_long = any(
                call.error is not None
                and "maximum context length" in call.error.message
                for call in trace.calls
            )
            if first_prompt_too_long:
                logger.warning(
                    "rollout group=%d/rollout=%d: the first prompt of task %s does not fit the "
                    "context; marking %s",
                    group_id,
                    rollout_id,
                    trace.task.key,
                    status.name,
                )

        # ======== Record the reward and logs ========
        # Log Verifiers' named rewards and metrics on turn 0, e.g. verifiers/reward/solved.
        # - Every rollout with turns logs them, whatever its status.
        # - A reward is its raw score; the training reward, `trace.reward`, is their weighted sum.
        # - None means Verifiers never computed it (e.g. scoring didn't run): skip it.
        if turns:
            turns[0].metrics.extend(
                m.Metric(f"verifiers/reward/{name}", m.Mean(reward.score))
                for name, reward in trace.rewards.items()
                if reward is not None
            )
            turns[0].metrics.extend(
                m.Metric(f"verifiers/metric/{name}", m.Mean(value))
                for name, value in trace.metrics.items()
                if value is not None
            )
        logs = verifiers_rollout_logs(verifiers_episode=verifiers_episode, trace=trace)
        if status.is_error():
            log_failed_rollout(logs=logs, group_id=group_id, rollout_id=rollout_id)
        else:
            # TODO: give the rubric the rollout's Verifiers reward directly. It rides on the last
            # turn only because `RewardFromVerifiers` reads rewards from turns. Blocked: it changes
            # `RewardFn`'s input for both workers.
            turns[-1].env_rewards[VERIFIERS_REWARD_KEY] = trace.reward
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
            logs=logs,
        )

    @staticmethod
    def rollout_status(
        *, verifiers_episode: vf.Episode, trace: vf.Trace
    ) -> RolloutStatus:
        """Say who failed: the policy (trained with `error_reward`) or something outside it (not
        trained).

            trace                                   status
            setup_timeout                           ERROR_INFRA   the sandbox or harness didn't start in time
            another deadline (agent, scoring, ...)  ERROR_TIMEOUT
            failed, last error HarnessError         ERROR         e.g. the agent's `pkill -f` killed its harness
            failed, last error TaskError            ERROR         e.g. the agent broke its box: no reward file
            failed, last error ProviderError 400    ERROR         the prompt outgrew the context
            failed, any other error or none         ERROR_INFRA   e.g. SandboxError, TunnelError, a 502
            ok, but the episode failed elsewhere    ERROR_INFRA   e.g. another agent's trace, an env hook
            ok                                      COMPLETED / TRUNCATED_*

        `_run_single_rollout` then makes any trainable status ERROR_INFRA for a rollout with no turn.
        """
        # TODO: tell policy-caused failures from infra ones inside HarnessError, TaskError and the
        #   deadline errors (agent, finalize, scoring). Today they all train: e.g. Harbor turns a
        #   sandbox error while reading the reward file into a TaskError. Blocked: Verifiers' error
        #   types say where a rollout broke, not who caused it.
        if trace.stop_condition == "setup_timeout":
            return RolloutStatus.ERROR_INFRA
        # Verifiers flags an expired deadline with is_timeout: a stage's on its trace (which can keep
        # ok=True), the whole episode's on the episode.
        if trace.is_timeout or verifiers_episode.is_timeout:
            return RolloutStatus.ERROR_TIMEOUT
        if not trace.ok:
            # The newest error: the one that ended this trace, after any retried attempt's.
            error = trace.last_error
            policy_can_cause = error is not None and (
                error.type in ("HarnessError", "TaskError")
                or (error.type == "ProviderError" and error.status_code == 400)
            )
            return (
                RolloutStatus.ERROR if policy_can_cause else RolloutStatus.ERROR_INFRA
            )
        if not verifiers_episode.ok:
            return RolloutStatus.ERROR_INFRA
        if not trace.is_truncated:
            return RolloutStatus.COMPLETED
        if trace.stop_condition == "max_turns":
            return RolloutStatus.TRUNCATED_MAX_TURNS
        return RolloutStatus.TRUNCATED_LENGTH

    @staticmethod
    def trace_to_rollout_turns(
        *,
        trace: vf.Trace,
        generation_metadata: VerifiersGenerationMetadata | None,
        group_id: int,
        rollout_id: int,
    ) -> list[RolloutTurn]:
        """Flatten a Verifiers trace into TitanRL rollout turns, one per span of trainable tokens.

        Example, one branch: user [1, 2], sampled [3, 4], tool result [5, 6], sampled [7, 8].
        A turn stores its prompt as the length it shares with the previous turn's prompt +
        completion (`prompt_prefix_len`), plus the new tokens (`prompt_delta_token_ids`):

            turn 0   prefix 0, delta [1, 2]   completion [3, 4]   env_messages [tool result]
            turn 1   prefix 4, delta [5, 6]   completion [7, 8]

        A node shared by several branches is emitted once. Every turn gets the rollout-wide
        min/max policy version from the generation server; generator metrics go on turn 0 only.
        """
        if generation_metadata is None:
            if any(any(node.mask) for node in trace.nodes):
                raise ValueError(
                    "Verifiers trace has trainable tokens but no generation metadata"
                )
            return []

        node_index = {id(node): index for index, node in enumerate(trace.nodes)}
        trained_nodes: set[int] = set()
        replied_nodes: set[int] = set()
        last_turn_by_node: dict[int, RolloutTurn] = {}
        turns: list[RolloutTurn] = []
        # The last appended turn's prompt + completion; not reset between branches
        previous_token_ids: list[int] = []

        for branch in trace.branches:
            token_ids = branch.token_ids
            logprobs = branch.logprobs
            branch_offset = 0
            reply_to: RolloutTurn | None = None
            for node in branch.nodes:
                index = node_index[id(node)]
                mask = list(node.mask)
                if node.sampled and any(mask):
                    if index in trained_nodes:
                        mask = [False] * len(mask)
                    else:
                        trained_nodes.add(index)
                for start, end in _trainable_token_spans(mask):
                    absolute_start = branch_offset + start
                    absolute_end = branch_offset + end
                    prompt_prefix_len, prompt_delta_token_ids = split_prompt(
                        Prompt(token_ids=token_ids[:absolute_start]),
                        previous=Prompt(token_ids=previous_token_ids),
                    )
                    previous_token_ids = token_ids[:absolute_end]
                    turns.append(
                        RolloutTurn(
                            rollout_id=RolloutTurnID(
                                group_id=group_id,
                                rollout_id=rollout_id,
                                turn_id=len(turns),
                            ),
                            prompt_prefix_len=prompt_prefix_len,
                            prompt_delta_token_ids=prompt_delta_token_ids,
                            completion_token_ids=np.asarray(
                                token_ids[absolute_start:absolute_end], dtype=np.int32
                            ),
                            completion_logprobs=np.asarray(
                                logprobs[absolute_start:absolute_end], dtype=np.float32
                            ),
                            min_policy_version=generation_metadata.min_policy_version,
                            max_policy_version=generation_metadata.max_policy_version,
                            completion_message=message_to_wire(node.message),
                            metrics=(
                                list(generation_metadata.metrics) if not turns else []
                            ),
                        )
                    )
                    last_turn_by_node[index] = turns[-1]
                if node.sampled:
                    reply_to = last_turn_by_node.get(index)
                elif reply_to is not None and index not in replied_nodes:
                    replied_nodes.add(index)
                    reply_to.env_messages.append(message_to_wire(node.message))
                branch_offset += len(node.token_ids)
        return turns


def _trainable_token_spans(mask: list[bool]) -> list[tuple[int, int]]:
    """Return half-open token spans marked trainable by a Verifiers node mask."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, sampled in enumerate([*mask, False]):
        if sampled and start is None:
            start = index
        elif not sampled and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _local_taskset_module(taskset: VerifiersTasksetConfig) -> str | None:
    """Return the module backing a locally registered taskset alias."""
    module = type(taskset).__module__
    alias = module.replace(".", "_").lower()
    return module if taskset.id == alias else None


def verifiers_rollout_logs(
    verifiers_episode: vf.Episode, trace: vf.Trace
) -> dict[str, Any]:
    """Summarize how a Verifiers rollout ended, for ``Rollout.logs``.

    Verifiers' own "rollout done" line prints only the error type or stop
    condition. This keeps the error messages and tracebacks (a failed
    harness's message ends with its stderr), per-phase wall time and model-call
    latency. For example, ``agent_sec=7200 model_sec=6900
    slowest_model_call_sec=1801`` on an agent timeout points at slow generation,
    not the task.
    """
    timing = trace.timing
    call_seconds = [call.time.duration for call in trace.calls]
    return {
        "verifiers_trace_id": trace.id,
        "task": trace.task.key,
        "stop_condition": trace.stop_condition,
        "errors": [
            error.model_dump(mode="json")
            for error in (*verifiers_episode.errors, *trace.errors)
        ],
        "setup_sec": timing.setup.duration,
        "agent_sec": timing.agent.duration,
        "model_sec": timing.agent.model.duration,
        "harness_sec": timing.agent.harness.duration,
        "scoring_sec": timing.scoring.duration,
        "model_calls": len(call_seconds),
        "failed_model_calls": sum(call.error is not None for call in trace.calls),
        "slowest_model_call_sec": max(call_seconds, default=0.0),
    }


def log_failed_rollout(logs: dict[str, Any], *, group_id: int, rollout_id: int) -> None:
    """Log the last error and timing from ``verifiers_rollout_logs`` in one line."""
    error = logs["errors"][-1] if logs["errors"] else {}
    traceback_tail = " | ".join(
        (error.get("traceback") or "").strip().splitlines()[-3:]
    )
    logger.warning(
        "Verifiers rollout failed: trace=%s group=%d rollout=%d task=%s stop=%s "
        "error=%s: %s | setup=%.0fs agent=%.0fs (model=%.0fs harness=%.0fs) "
        "scoring=%.0fs | model_calls=%d failed_calls=%d slowest_call=%.0fs | %s",
        logs["verifiers_trace_id"],
        group_id,
        rollout_id,
        logs["task"],
        logs["stop_condition"],
        error.get("type"),
        error.get("message"),
        logs["setup_sec"],
        logs["agent_sec"],
        logs["model_sec"],
        logs["harness_sec"],
        logs["scoring_sec"],
        logs["model_calls"],
        logs["failed_model_calls"],
        logs["slowest_model_call_sec"],
        traceback_tail,
    )
