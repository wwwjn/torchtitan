# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ulysses CUDA-graph registration policy before process-group initialization."""

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from torchtitan.config import Configurable
from torchtitan.distributed.context_parallel import (
    configure_ulysses_nccl_graph_registration,
)
from torchtitan.models.common.attention import FlexInnerAttention, InnerAttention
from torchtitan.models.common.attention.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
    UlyssesCPSlidingWindowFlexInnerAttention,
    UlyssesCPVarlenInnerAttention,
)
from torchtitan.protocols import BaseModel
from torchtitan.training_engine import TrainingEngine


@dataclass(kw_only=True, slots=True)
class _AttentionConfigTree(Configurable.Config):
    inner_attentions: list[InnerAttention.Config]


@pytest.fixture(autouse=True)
def _unset_graph_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_GRAPH_REGISTER", raising=False)


@pytest.mark.parametrize(
    "attention_cls",
    [
        UlyssesCPFlexInnerAttention,
        UlyssesCPSlidingWindowFlexInnerAttention,
        UlyssesCPVarlenInnerAttention,
    ],
)
@pytest.mark.parametrize("all_to_all_backend", ["list", "single"])
@pytest.mark.parametrize("nccl_version", [(2, 30, 7), (2, 32, 2)])
def test_older_nccl_disables_registration_for_ulysses(
    attention_cls: type,
    all_to_all_backend: str,
    nccl_version: tuple[int, int, int],
) -> None:
    attention_config = (
        attention_cls.Config(all_to_all_backend=all_to_all_backend)
        if attention_cls is not UlyssesCPSlidingWindowFlexInnerAttention
        else attention_cls.Config(
            all_to_all_backend=all_to_all_backend, window_size=128
        )
    )
    model_config = _AttentionConfigTree(
        inner_attentions=[FlexInnerAttention.Config(), attention_config]
    )
    with (
        patch("torch.cuda.nccl.version", return_value=nccl_version),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        configure_ulysses_nccl_graph_registration(model_config)
    assert os.environ["NCCL_GRAPH_REGISTER"] == "0"


@pytest.mark.parametrize("nccl_version", [(2, 32, 3), (2, 33, 0)])
@pytest.mark.parametrize("existing_value", [None, "0", "1"])
def test_fixed_nccl_preserves_registration_policy(
    monkeypatch: pytest.MonkeyPatch,
    nccl_version: tuple[int, int, int],
    existing_value: str | None,
) -> None:
    if existing_value is not None:
        monkeypatch.setenv("NCCL_GRAPH_REGISTER", existing_value)
    with patch("torch.cuda.nccl.version", return_value=nccl_version):
        configure_ulysses_nccl_graph_registration(
            UlyssesCPVarlenInnerAttention.Config()
        )
    assert os.environ.get("NCCL_GRAPH_REGISTER") == existing_value


def test_older_nccl_overrides_unsafe_explicit_setting(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("NCCL_GRAPH_REGISTER", "1")
    with (
        patch("torch.cuda.nccl.version", return_value=(2, 30, 7)),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        configure_ulysses_nccl_graph_registration(
            UlyssesCPVarlenInnerAttention.Config()
        )
    assert os.environ["NCCL_GRAPH_REGISTER"] == "0"
    assert "Setting NCCL_GRAPH_REGISTER=0" in caplog.text


def test_already_disabled_registration_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("NCCL_GRAPH_REGISTER", "0")
    with (
        patch("torch.cuda.nccl.version", return_value=(2, 30, 7)),
        patch("torch.distributed.is_initialized", return_value=True),
    ):
        configure_ulysses_nccl_graph_registration(
            UlyssesCPVarlenInnerAttention.Config()
        )
    assert os.environ["NCCL_GRAPH_REGISTER"] == "0"
    assert not caplog.records


def test_required_setting_rejects_late_initialization() -> None:
    with (
        patch("torch.cuda.nccl.version", return_value=(2, 30, 7)),
        patch("torch.distributed.is_initialized", return_value=True),
        pytest.raises(RuntimeError, match="before process-group initialization"),
    ):
        configure_ulysses_nccl_graph_registration(
            UlyssesCPVarlenInnerAttention.Config()
        )
    assert "NCCL_GRAPH_REGISTER" not in os.environ


@pytest.mark.parametrize(
    "attention_config",
    [FlexInnerAttention.Config(), KVAllGatherCPFlexInnerAttention.Config()],
)
def test_other_attention_backends_do_not_change_registration(
    attention_config: InnerAttention.Config,
) -> None:
    with patch("torch.cuda.nccl.version") as nccl_version:
        configure_ulysses_nccl_graph_registration(attention_config)
    nccl_version.assert_not_called()
    assert "NCCL_GRAPH_REGISTER" not in os.environ


@pytest.mark.parametrize(
    (
        "cp_degree",
        "disable_cuda_graphs",
        "comm_backend",
        "device_type",
        "hip",
        "has_nccl",
        "expected",
    ),
    [
        (2, False, "default", "cuda", None, True, "0"),
        (1, False, "default", "cuda", None, True, None),
        (2, True, "default", "cuda", None, True, None),
        (2, False, "fake", "cuda", None, True, None),
        (2, False, "real_pp_fake_spmd", "cuda", None, True, None),
        (2, False, "default", "cpu", None, True, None),
        (2, False, "default", "cuda", "7.0", True, None),
        (2, False, "default", "cuda", None, False, None),
    ],
)
def test_engine_sets_policy_before_subclass_process_group_hook(
    cp_degree: int,
    disable_cuda_graphs: bool,
    comm_backend: str,
    device_type: str,
    hip: str | None,
    has_nccl: bool,
    expected: str | None,
) -> None:
    class _EngineWithRuntimeHook(TrainingEngine):
        def _initialize_distributed_runtime(self) -> None:
            assert os.environ.get("NCCL_GRAPH_REGISTER") == expected

    config = SimpleNamespace(
        parallelism=SimpleNamespace(context_parallel_degree=cp_degree),
        training=SimpleNamespace(disable_cuda_graphs=disable_cuda_graphs),
        comm=SimpleNamespace(backend=comm_backend),
    )
    with (
        patch("torchtitan.training_engine.utils.device_type", device_type),
        patch("torch.version.hip", hip),
        patch("torch.distributed.is_nccl_available", return_value=has_nccl),
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.cuda.nccl.version", return_value=(2, 30, 7)),
        patch("torchtitan.training_engine.has_quantization", return_value=False),
    ):
        _EngineWithRuntimeHook(
            cast(TrainingEngine.Config, config),
            model_config=cast(BaseModel.Config, UlyssesCPVarlenInnerAttention.Config()),
            max_num_documents=None,
            output_dir="",
        )
    assert os.environ.get("NCCL_GRAPH_REGISTER") == expected
