# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sizing the Verifiers env-server pool from the controller's rollouts in flight."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.rl.examples.verifiers.env_server import size_serve_config
from torchtitan_recipes.rl.verifiers_dapo_math import _verifiers_math_rollouter_config

ADDRESS = "tcp://127.0.0.1:0"


def test_unset_pool_is_sized_from_rollouts_in_flight() -> None:
    serve = size_serve_config(
        vf.ServeConfig(address=ADDRESS),
        rollout_concurrency=1024,
        rollouts_per_worker=8,
    )

    assert serve.pool == vf.StaticPoolConfig(num_workers=128)
    assert serve.max_concurrent == 8
    assert serve.address == ADDRESS


def test_fewer_rollouts_than_one_worker_holds_use_one_worker() -> None:
    serve = size_serve_config(
        vf.ServeConfig(address=ADDRESS),
        rollout_concurrency=5,
        rollouts_per_worker=8,
    )

    assert serve.pool == vf.StaticPoolConfig(num_workers=1)
    assert serve.max_concurrent == 5


def test_pinned_pool_is_kept_and_warns_when_too_small(caplog) -> None:
    pinned = vf.ServeConfig(
        pool=vf.StaticPoolConfig(num_workers=16), max_concurrent=2, address=ADDRESS
    )

    with caplog.at_level(logging.WARNING):
        serve = size_serve_config(pinned, rollout_concurrency=64, rollouts_per_worker=8)

    assert serve == pinned
    assert "at most 32 rollouts" in caplog.text
    assert "keeps 64 in flight" in caplog.text


def test_pinned_pool_without_a_per_worker_bound_does_not_warn(caplog) -> None:
    pinned = vf.ServeConfig(pool=vf.StaticPoolConfig(num_workers=1), address=ADDRESS)

    with caplog.at_level(logging.WARNING):
        serve = size_serve_config(
            pinned, rollout_concurrency=1024, rollouts_per_worker=8
        )

    assert serve == pinned
    assert not caplog.records


def test_rollouts_per_env_worker_must_be_positive() -> None:
    config = _verifiers_math_rollouter_config(max_rollout_tokens=1024)

    with pytest.raises(ValueError, match="rollouts_per_env_worker"):
        replace(config, rollouts_per_env_worker=0)
