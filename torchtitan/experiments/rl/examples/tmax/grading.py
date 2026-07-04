# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""tmax grading: run the task's verifier IN the sandbox and read back the reward.

The tmax verifier contract (AI2 terminal-agent tasks): run ``bash /tests/test.sh``
INSIDE the task container; the script writes ``/logs/verifier/reward.txt`` holding
``0`` or ``1``. Reward = that value.

Unlike R2E (which re-boots a CLEAN eval sandbox and applies a git diff), tmax has
no diff step: the agent mutates the container's filesystem directly and the
verifier inspects that same filesystem. So ``TMaxRollouter`` KEEPS the agent's
sandbox and grades in place -- this module uploads the verifier + fixtures into
the live sandbox, runs it, and parses the reward file.

Two entry points, both driving the same steps:
  - ``grade_tmax(sb, tmax, workdir, ...)`` -- for the harness ``Sandbox`` contract
    (async ``exec`` / ``write_file`` / ``read_file``); used by the rollouter.
  - ``grade_tmax_daytona(sb, tmax, workdir, ...)`` -- for a RAW ``daytona`` Sandbox
    (sync ``process.exec`` / ``fs.upload_file``); used by ``local_smoke.py`` so the
    grading logic can be exercised without the full training stack.
"""

from __future__ import annotations

import logging
import os
import posixpath
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only so this module imports WITHOUT the torchtitan/vLLM stack -- the
    # standalone ``local_smoke.py`` (daytona-only venv) imports it just to call
    # ``grade_tmax_daytona`` + the pure helpers, which have no torchtitan dependency.
    from torchtitan.experiments.rl.harness import Sandbox

logger = logging.getLogger(__name__)

# In-sandbox layout the verifier contract fixes.
_TESTS_DIR = "/tests"
_TEST_SH = "/tests/test.sh"
_VERIFIER_DIR = "/logs/verifier"
_DEFAULT_REWARD_PATH = "/logs/verifier/reward.txt"

# environment/seeds/<rel> fixtures seed the agent's workspace (<workdir>/<rel>);
# tests/<rel> fixtures live next to test.sh (/tests/<rel>).
_SEEDS_PREFIX = "environment/seeds/"
_TESTS_PREFIX = "tests/"


def _eval_timeout_sec() -> int:
    val = os.environ.get("TMAX_EVAL_TIMEOUT_SEC")
    return int(val) if val and val.strip() else 900


def _fixture_dest(rel: str, workdir: str) -> str | None:
    """Map a fixture relpath to its in-sandbox destination.

    ``environment/seeds/aa`` -> ``<workdir>/aa``;
    ``tests/expected_output.txt`` -> ``/tests/expected_output.txt``.
    Anything else (unexpected layout) is skipped.
    """
    rel = rel.lstrip("/")
    if rel == "tests/test.sh":
        return None  # test.sh is uploaded explicitly, never as a fixture
    if rel.startswith(_SEEDS_PREFIX):
        sub = rel[len(_SEEDS_PREFIX) :]
        return posixpath.join(workdir, sub) if sub else None
    if rel.startswith(_TESTS_PREFIX):
        sub = rel[len(_TESTS_PREFIX) :]
        return posixpath.join(_TESTS_DIR, sub) if sub else None
    return None


def _parse_reward(text: str) -> float:
    """Parse ``reward.txt`` contents into a float in [0, 1] (0.0 if unparseable)."""
    try:
        val = float((text or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0
    return max(0.0, min(1.0, val))


# --------------------------------------------------------------------------- #
# Harness Sandbox path (async) -- used by the rollouter.
# --------------------------------------------------------------------------- #
async def grade_tmax(
    sb: Sandbox,
    tmax: dict,
    *,
    workdir: str,
    timeout_sec: int | None = None,
) -> float:
    """Grade a tmax task in the (already-run) sandbox ``sb`` and return reward.

    Creates ``/logs/verifier`` + ``/tests``, uploads ``test.sh`` and every
    fixture to its destination, runs ``bash /tests/test.sh``, then reads back
    ``reward_path`` (default ``/logs/verifier/reward.txt``). Returns a float in
    [0, 1]; 0.0 when the reward file is missing or unparseable.
    """
    timeout = timeout_sec if timeout_sec is not None else _eval_timeout_sec()
    test_sh = tmax.get("test_sh") or ""
    fixtures = tmax.get("fixtures") or {}
    reward_path = tmax.get("reward_path") or _DEFAULT_REWARD_PATH

    await sb.exec(
        f"mkdir -p {shlex.quote(_VERIFIER_DIR)} {shlex.quote(_TESTS_DIR)} "
        f"{shlex.quote(workdir)}",
        user="root",
        check=False,
        timeout=60,
    )
    for rel, content in fixtures.items():
        dest = _fixture_dest(rel, workdir)
        if dest is None:
            continue
        await sb.write_file(dest, content, user="root")
    await sb.write_file(_TEST_SH, test_sh, user="root")

    # test.sh scripts assume they are invoked as `bash /tests/test.sh` (they use
    # $(dirname "$0") to find sibling fixtures). Run as root so /logs and any
    # system path is writable; the verifier is trusted dataset code.
    await sb.exec(
        f"chmod +x {shlex.quote(_TEST_SH)}; bash {shlex.quote(_TEST_SH)}",
        user="root",
        check=False,
        timeout=timeout,
    )
    reward_txt = await sb.read_file(reward_path, user="root")
    reward = _parse_reward(reward_txt)
    logger.info("[tmax] graded reward=%.2f (reward_path=%s)", reward, reward_path)
    return reward


# --------------------------------------------------------------------------- #
# Raw daytona Sandbox path (sync) -- used by local_smoke.py.
# --------------------------------------------------------------------------- #
def grade_tmax_daytona(
    sb,
    tmax: dict,
    *,
    workdir: str,
    timeout_sec: int | None = None,
) -> float:
    """Same steps as ``grade_tmax`` but against a RAW ``daytona`` Sandbox.

    Uses the daytona SDK's sync API directly (``sb.process.exec`` /
    ``sb.fs.upload_file``) so the grading logic can be exercised in a standalone
    script without importing the full torchtitan/vLLM training stack.
    """
    timeout = timeout_sec if timeout_sec is not None else _eval_timeout_sec()
    test_sh = tmax.get("test_sh") or ""
    fixtures = tmax.get("fixtures") or {}
    reward_path = tmax.get("reward_path") or _DEFAULT_REWARD_PATH

    sb.process.exec(
        f"mkdir -p {shlex.quote(_VERIFIER_DIR)} {shlex.quote(_TESTS_DIR)} "
        f"{shlex.quote(workdir)}",
        timeout=60,
    )
    for rel, content in fixtures.items():
        dest = _fixture_dest(rel, workdir)
        if dest is None:
            continue
        sb.fs.upload_file(content.encode("utf-8"), dest)
    sb.fs.upload_file(test_sh.encode("utf-8"), _TEST_SH)

    sb.process.exec(
        f"chmod +x {shlex.quote(_TEST_SH)}; bash {shlex.quote(_TEST_SH)}",
        timeout=timeout,
    )
    r = sb.process.exec(f"cat {shlex.quote(reward_path)}", timeout=30)
    reward_txt = r.result if getattr(r, "exit_code", 1) == 0 else ""
    return _parse_reward(reward_txt)
