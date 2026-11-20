# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Each trainer DP replica's leader packs its column of the step plan and broadcasts it to the
replica's other ranks (`MicrobatchGroupReader`): every rank must train on exactly the
microbatches the batcher used to pack for its column."""

from __future__ import annotations

import contextvars
import dataclasses
import threading
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.rollout.finished_group import (
    FinishedRolloutGroup,
    join_rollout_group,
    split_rollout_group,
)
from torchtitan.rl.rollout.types import RolloutGroup
from torchtitan.rl.trainer.batcher import Batcher, MicrobatchPacker, TrainingMicrobatch
from torchtitan.rl.trainer.microbatch_group_reader import (
    _NUM_MICROBATCH_GROUPS_READ_AHEAD,
    MicrobatchGroupReader,
    ReplicaColumn,
)
from torchtitan.rl.trainer.trainer import trainer_ranks_by_dp_rank
from torchtitan.rl.trainer.training_sample_builder import TrainingSampleBuilder

from tests.rl.unit_tests.cpu.test_training_sample_spec import _NAN, _rollout

_SEQ_LEN = 16
_NUM_TOKENS_PER_MICROBATCH = 32


def _groups() -> list[RolloutGroup]:
    """4 text groups of 4 rollouts: varied lengths, NaN logprobs, a two-turn sample and a
    history edit (two samples of one rollout)."""
    groups = []
    for group_id in range(4):
        rollouts = [
            _rollout(
                0,
                [([1, 2], [-0.1] * (3 + group_id))],
                group_id=group_id,
                reward=0.0,
            ),
            _rollout(
                1,
                [([1], [_NAN, -0.2, -0.3])],
                group_id=group_id,
                reward=1.0,
            ),
            _rollout(
                2,
                [([1, 2, 3], [-0.1, -0.2]), ([1, 2, 3, 200, 201, 9], [-0.1, _NAN])],
                group_id=group_id,
                reward=2.0,
            ),
            _rollout(
                3,
                [([1, 2], [-0.1]), ([9, 9, 9], [-0.1, -0.2, -0.3])],
                group_id=group_id,
                reward=3.0,
            ),
        ]
        groups.append(RolloutGroup(group_id=group_id, rollouts=rollouts))
    return groups


def _packer() -> MicrobatchPacker:
    return MicrobatchPacker(
        num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
        max_context_length=_SEQ_LEN,
        pad_id=0,
        temperature=1.0,
        per_sample_pad_multiple=1,
        num_mtp_layers=0,
        build_mrope_positions=False,
        vision_processor=None,
        num_topk_logprobs=0,
    )


def _plan_columns(
    *, dp_degree: int, num_pp_microbatches: int
) -> tuple[list[ReplicaColumn], dict[int, FinishedRolloutGroup]]:
    """Plan `_groups()` with the batcher, and cut each DP rank's column as the controller does."""
    batcher = Batcher.Config().build(
        num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
        max_context_length=_SEQ_LEN,
        num_prompts_per_train_step=4,
        dp_degree=dp_degree,
        num_pp_microbatches=num_pp_microbatches,
    )
    finished_by_id = {
        group.group_id: FinishedRolloutGroup(*split_rollout_group(group))
        for group in _groups()
    }
    # The 4th group completes the step.
    [*_, (batch, _)] = [
        batcher.add_group(summary=finished.summary)
        for finished in finished_by_id.values()
    ]
    assert batch is not None
    controller = SimpleNamespace(
        trainer_dp_degree=dp_degree, _finished_groups_by_id=finished_by_id
    )
    return Controller._replica_columns(controller, batch.plan), finished_by_id


def _batcher_packed_column(
    column: ReplicaColumn, finished_by_id: dict[int, FinishedRolloutGroup]
) -> list[list[TrainingMicrobatch]]:
    """The column as the batcher packed it before leaders did: every sample materialized from
    the full groups, then each microbatch packed in plan order."""
    samples = iter(
        TrainingSampleBuilder.Config()
        .build()
        .materialize(
            training_sample_specs=[
                spec
                for group in column.microbatch_groups
                for specs in group
                for spec in specs
            ],
            rollout_groups={
                group_id: join_rollout_group(finished.summary, finished.arrays)
                for group_id, finished in finished_by_id.items()
            },
        )
    )
    packer = _packer()
    return [
        [packer.pack([next(samples) for _ in specs]) for specs in group]
        for group in column.microbatch_groups
    ]


def _assert_microbatches_equal(
    actual: list[list[TrainingMicrobatch]], expected: list[list[TrainingMicrobatch]]
) -> None:
    assert [len(group) for group in actual] == [len(group) for group in expected]
    for actual_group, expected_group in zip(actual, expected, strict=True):
        for actual_microbatch, expected_microbatch in zip(
            actual_group, expected_group, strict=True
        ):
            torch.testing.assert_close(
                dataclasses.asdict(actual_microbatch),
                dataclasses.asdict(expected_microbatch),
                rtol=0,
                atol=0,
                equal_nan=True,
            )


def _run_rank(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
    layout: dict[str, int],
    num_pp_microbatches: int,
    break_leader_column: bool,
) -> None:
    """One trainer rank: join its DP replica's gloo group, read its microbatch groups, and save
    them (or the error it raised) to `result_dir`."""
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        ranks_by_dp_rank = trainer_ranks_by_dp_rank(**layout)
        replica_group, _ = dist.new_subgroups_by_enumeration(
            ranks_by_dp_rank, backend="gloo"
        )
        dp_rank = next(
            dp_rank for dp_rank, ranks in enumerate(ranks_by_dp_rank) if rank in ranks
        )
        columns, _ = _plan_columns(
            dp_degree=layout["dp_degree"], num_pp_microbatches=num_pp_microbatches
        )
        column = columns[dp_rank] if ranks_by_dp_rank[dp_rank][0] == rank else None
        if column is not None and break_leader_column:
            # Without its rollouts, materialize raises on the leader's first group.
            column = dataclasses.replace(column, rollout_groups=[])
        reader = MicrobatchGroupReader(
            num_microbatch_groups=len(columns[dp_rank].microbatch_groups),
            replica_group=replica_group,
            column=column,
            training_sample_builder=TrainingSampleBuilder.Config().build(),
            microbatch_packer=_packer(),
        )
        try:
            microbatch_groups = list(reader)
        except Exception as error:
            Path(result_dir, f"{rank}.error").write_text(
                f"{type(error).__name__}: {error}"
            )
        else:
            torch.save(microbatch_groups, Path(result_dir, f"{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn_ranks(
    tmp_path: Path,
    *,
    layout: dict[str, int],
    num_pp_microbatches: int = 1,
    break_leader_column: bool = False,
) -> int:
    world_size = (
        layout["pp_degree"]
        * layout["dp_degree"]
        * layout["cp_degree"]
        * layout["tp_degree"]
    )
    mp.spawn(
        _run_rank,
        args=(
            world_size,
            str(tmp_path / "init"),
            str(tmp_path),
            layout,
            num_pp_microbatches,
            break_leader_column,
        ),
        nprocs=world_size,
        join=True,
    )
    return world_size


@pytest.mark.parametrize(
    ("layout", "num_pp_microbatches"),
    [
        (dict(pp_degree=1, dp_degree=2, cp_degree=1, tp_degree=2), 1),
        (dict(pp_degree=2, dp_degree=1, cp_degree=1, tp_degree=2), 2),
    ],
    ids=["dp2_tp2", "pp2_tp2"],
)
def test_every_rank_trains_on_the_microbatches_the_batcher_packed(
    tmp_path: Path, layout: dict[str, int], num_pp_microbatches: int
) -> None:
    world_size = _spawn_ranks(
        tmp_path, layout=layout, num_pp_microbatches=num_pp_microbatches
    )

    columns, finished_by_id = _plan_columns(
        dp_degree=layout["dp_degree"], num_pp_microbatches=num_pp_microbatches
    )
    for dp_rank, ranks in enumerate(trainer_ranks_by_dp_rank(**layout)):
        expected = _batcher_packed_column(columns[dp_rank], finished_by_id)
        assert all(len(group) == num_pp_microbatches for group in expected)
        for rank in ranks:
            _assert_microbatches_equal(
                torch.load(tmp_path / f"{rank}.pt", weights_only=False), expected
            )
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == sorted(
        f"{rank}.pt" for rank in range(world_size)
    )


def test_a_leader_that_fails_to_pack_fails_every_rank_of_its_replica(
    tmp_path: Path,
) -> None:
    start = time.monotonic()
    _spawn_ranks(
        tmp_path,
        layout=dict(pp_degree=1, dp_degree=2, cp_degree=1, tp_degree=2),
        break_leader_column=True,
    )

    # Within seconds, not the 60 s gloo timeout.
    assert time.monotonic() - start < 50
    for rank in range(4):
        error = (tmp_path / f"{rank}.error").read_text()
        if rank in (0, 2):  # the leaders
            assert error.startswith("KeyError")
        else:
            assert error.startswith("RuntimeError: The DP replica's leader failed")
            assert "KeyError" in error


def _counting_reader(
    num_groups: int, packed: list[int], *, pack=None
) -> MicrobatchGroupReader:
    """A reader on a replica of one rank whose packer records each group it packs."""

    def default_pack(samples: list[str]) -> SimpleNamespace:
        [index] = samples
        packed.append(index)
        return SimpleNamespace(preprocess_images=lambda: index)

    return MicrobatchGroupReader(
        num_microbatch_groups=num_groups,
        replica_group=None,
        column=ReplicaColumn(
            microbatch_groups=[[[index]] for index in range(num_groups)],
            rollout_groups=[],
        ),
        training_sample_builder=SimpleNamespace(
            materialize=lambda *, training_sample_specs, rollout_groups: list(
                training_sample_specs
            )
        ),
        microbatch_packer=SimpleNamespace(pack=pack or default_pack),
    )


def test_reader_yields_in_order_and_packs_two_groups_ahead() -> None:
    packed: list[int] = []
    reader = _counting_reader(6, packed)

    assert len(reader) == 6
    for [index] in reader:
        # Wait for the expected group, then 20 ms more, so a group packed too far ahead shows.
        expected = min(index + _NUM_MICROBATCH_GROUPS_READ_AHEAD, 5)
        deadline = time.monotonic() + 5
        while max(packed) < expected and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.02)
        assert max(packed) == expected
    assert packed == list(range(6))
    # A second pass (e.g. an SDC replay of the step) packs the column again.
    assert list(reader) == [[index] for index in range(6)]


def test_reader_thread_sees_the_callers_context() -> None:
    step = contextvars.ContextVar("step")
    step.set(7)
    threads = []

    def pack(samples: list[int]) -> SimpleNamespace:
        threads.append(threading.current_thread())
        return SimpleNamespace(preprocess_images=lambda: step.get())

    assert list(_counting_reader(1, [], pack=pack)) == [[7]]
    assert threads[0] is not threading.main_thread()


def test_reader_skips_queued_groups_when_training_stops() -> None:
    packed: list[int] = []
    group_1_started, group_1_done = threading.Event(), threading.Event()

    def pack(samples: list[int]) -> SimpleNamespace:
        [index] = samples
        if index == 1:
            group_1_started.set()
            time.sleep(0.2)
            group_1_done.set()
        packed.append(index)
        return SimpleNamespace(preprocess_images=lambda: index)

    iterator = iter(_counting_reader(3, packed, pack=pack))

    assert next(iterator) == [0]
    assert group_1_started.wait(timeout=5)
    # Group 1 is packing and group 2 is queued: closing cancels group 2.
    iterator.close()
    assert group_1_done.wait(timeout=5)
    time.sleep(0.05)
    assert packed == [0, 1]
