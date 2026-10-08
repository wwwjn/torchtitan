# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for DistMuon-specific redistribution process groups."""

from unittest.mock import MagicMock, patch

import pytest
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.components.optim.optimizer import _DistMuonProcessGroupCache, DistMuon


def test_redistribution_max_ctas_is_opt_in() -> None:
    """Keep the existing storage-mesh communicator behavior by default."""
    config = DistMuon.Config(
        pattern=".*",
        compute_sharding_by_fqn={},
        bucket_configs=(),
    )

    assert config.redistribution_max_ctas is None


def test_redistribution_max_ctas_must_be_positive() -> None:
    """Reject invalid NCCL CTA limits before distributed initialization."""
    common = {
        "pattern": ".*",
        "compute_sharding_by_fqn": {},
        "bucket_configs": (),
    }
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            DistMuon.Config(redistribution_max_ctas=value, **common)


def test_process_group_cache_splits_once_with_independent_resources() -> None:
    """Reuse one CTA-capped child without modifying its parent group."""
    parent_group = MagicMock(spec=dist.ProcessGroup)
    child_group = MagicMock(spec=dist.ProcessGroup)
    mesh = MagicMock(spec=DeviceMesh)
    mesh.get_group.return_value = parent_group
    cache = _DistMuonProcessGroupCache()

    with (
        patch.object(dist, "get_backend", return_value=dist.Backend.NCCL),
        patch.object(dist, "get_world_size", return_value=64),
        patch.object(dist, "split_group", return_value=child_group) as split_group,
    ):
        first = cache.get_or_create(mesh, max_ctas=8)
        second = cache.get_or_create(mesh, max_ctas=8)

    assert first is child_group
    assert second is child_group
    split_group.assert_called_once()
    kwargs = split_group.call_args.kwargs
    assert kwargs["parent_pg"] is parent_group
    assert kwargs["split_ranks"] == [list(range(64))]
    assert kwargs["backend"] == "nccl"
    assert kwargs["pg_options"].config.max_ctas == 8
    assert kwargs["pg_options"].config.split_share == 0


def test_fake_process_group_preserves_metadata_validation(caplog) -> None:
    """Fake process groups retain the parent because they model no NCCL policy."""
    parent_group = MagicMock(spec=dist.ProcessGroup)
    mesh = MagicMock(spec=DeviceMesh)
    mesh.get_group.return_value = parent_group

    with patch.object(dist, "get_backend", return_value="fake"):
        resolved = _DistMuonProcessGroupCache().get_or_create(mesh, max_ctas=8)

    assert resolved is parent_group
    assert "not modeled by the fake process group" in caplog.text
