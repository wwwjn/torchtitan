# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""tmax terminal-agent dataset for the coding-agent RL example.

Reads a JSONL produced by ``prepare_tmax_data.py`` (R2E-compatible schema with a
``tmax`` metadata blob instead of ``r2e``). Each row::

    {
      "prompt": <instruction.md>,
      "label": <task_id>,
      "metadata": {
        "instance_id", "image" (docker.io/...), "workdir",
        "problem_statement": <instruction.md>,
        "tmax": {"test_sh", "fixtures": {relpath: content}, "reward_path"}
      }
    }

The dataset is an endless, seeded stream of frozen ``TMaxSample``s, mirroring
``SWER2EDataset`` (same Configurable interface: ``data_path`` / ``seed`` /
``shuffle`` config, ``__iter__`` / ``__next__``, ``state_dict`` /
``load_state_dict``).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass, field

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True, slots=True)
class TMaxSample:
    """One tmax terminal-agent task: a containerized env, an instruction, and a
    verifier script that writes a 0/1 reward inside the container."""

    instance_id: str
    """Stable task id (e.g. ``task_000000_c19dda5b``)."""

    image: str
    """Public docker image the task runs in (e.g. ``docker.io/hamishi740/...``)."""

    workdir: str
    """Working directory inside the sandbox (best-guess; default ``/workspace``)."""

    problem_statement: str
    """The instruction the agent must satisfy (instruction.md)."""

    tmax: dict = field(default_factory=dict)
    """Grading payload: ``test_sh``, ``fixtures`` ({relpath: content}), ``reward_path``."""


class TMaxDataset(Configurable):
    """Endless, seeded stream of tmax terminal-agent samples loaded from a JSONL."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        data_path: str = ""
        """Path to the tmax JSONL file (required)."""

        seed: int = 42
        """Seed for the row-order shuffle."""

        shuffle: bool = True
        """Shuffle row order (reshuffling on each wrap). Set False for validation."""

        holdout_n: int = 0
        """Reserve the LAST ``holdout_n`` rows (file order) as a held-out validation slice,
        disjoint from training. 0 = no split (whole file). Both the train and validation
        instances must pass the same ``holdout_n`` so the split matches."""

        split: str = "train"
        """Which slice this instance serves: ``train`` (rows[:-holdout_n]) or ``validation``
        (rows[-holdout_n:]). Ignored when ``holdout_n == 0``."""

        skip_ids_path: str = ""
        """Optional path to a zero-std annotation file (``SWE_ZERO_STD_LOG`` output from a
        prior run). Every ``instance_id`` listed there is dropped at load, so prompts that
        gave no learning signal (all-pass or all-fail groups) are not sampled again. Empty
        = keep all rows. Reads JSONL rows ``{"instance_id": ...}`` or bare ids per line."""

    def __init__(self, config: Config) -> None:
        if not config.data_path:
            raise ValueError("TMaxDataset.Config.data_path is required")
        if config.split not in ("train", "validation"):
            raise ValueError(
                f"TMaxDataset.Config.split must be 'train' or 'validation', got {config.split!r}"
            )
        samples: list[TMaxSample] = []
        with open(config.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                md = row.get("metadata") or {}
                instance_id = (
                    md.get("instance_id")
                    or (row.get("label") if isinstance(row.get("label"), str) else None)
                    or "unknown"
                )
                image = md.get("image")
                tmax = md.get("tmax") or {}
                if not image or not tmax:
                    raise ValueError(
                        f"row {instance_id!r} missing image/tmax in metadata"
                    )
                samples.append(
                    TMaxSample(
                        instance_id=instance_id,
                        image=image,
                        workdir=md.get("workdir") or "/workspace",
                        problem_statement=md.get("problem_statement")
                        or _coerce_prompt(row.get("prompt")),
                        tmax=tmax,
                    )
                )
        if not samples:
            raise ValueError(f"no rows found in {config.data_path}")

        # Skip prompts annotated zero-std by a prior run (no learning signal). Applied
        # before the holdout split so both train and validation instances (same file,
        # same order) exclude the same ids and the split stays aligned.
        if config.skip_ids_path:
            skip_ids = _load_skip_ids(config.skip_ids_path)
            if skip_ids:
                kept = [s for s in samples if s.instance_id not in skip_ids]
                logger.info(
                    f"TMaxDataset: skipped {len(samples) - len(kept)} zero-std prompt(s) "
                    f"from {config.skip_ids_path} ({len(kept)}/{len(samples)} remain)"
                )
                samples = kept
                if not samples:
                    raise ValueError(
                        f"all rows filtered out by skip_ids_path={config.skip_ids_path}"
                    )

        # Held-out split: the last holdout_n rows (in file order) form the validation slice,
        # disjoint from the training slice, so periodic validation measures generalization
        # rather than training-set recall. Deterministic (file order), no separate file.
        if config.holdout_n > 0:
            if config.holdout_n >= len(samples):
                raise ValueError(
                    f"holdout_n={config.holdout_n} >= dataset size {len(samples)}"
                )
            samples = (
                samples[-config.holdout_n :]
                if config.split == "validation"
                else samples[: -config.holdout_n]
            )
        self._samples = samples

        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[TMaxSample]:
        return self

    def __next__(self) -> TMaxSample:
        if self._pos >= len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        idx = self._order[self._pos]
        self._pos += 1
        return self._samples[idx]

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]


def _load_skip_ids(path: str) -> set[str]:
    """Read instance_ids to skip from a zero-std annotation file.

    Accepts either JSONL rows ``{"instance_id": ...}`` (the ``SWE_ZERO_STD_LOG``
    format) or a bare ``instance_id`` per line. Missing file = empty set (a first
    run has nothing to skip yet).
    """
    ids: set[str] = set()
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    iid = (json.loads(line) or {}).get("instance_id")
                    if iid:
                        ids.add(iid)
                else:
                    ids.add(line)
    except FileNotFoundError:
        logger.warning(f"TMaxDataset: skip_ids_path {path} not found; skipping nothing")
    return ids


def _coerce_prompt(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str):
                    return content
    return ""
