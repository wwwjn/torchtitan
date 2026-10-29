"""AIPerf (the official AgentX harness) with a conversation-end signal: every request it marks as
its conversation's final turn (``RequestInfo.is_final_turn``, which AIPerf already uses to release
its own connection lease) also carries ``X-Session-Final-Turn: 1``, next to the per-conversation
``X-Correlation-ID``. TitanRL's server (``--release-on-final-turn``) then releases the session's
held KV when that turn ends, as the RL rollouter does when a rollout ends. Requests, timing and
dataset are unchanged; native vLLM ignores the header. WARMUP credits of trajectories active at
the profiling start are not final turns (AIPerf resumes them at the next turn), so they carry no
header.

Loaded through ``PYTHONPATH`` by ``run_agentx.sh`` (``HOLDER=1``); every
harness process (workers included) runs it at startup and patches the transport once it loads.
"""

import importlib.abc
import importlib.util
import sys

_TARGET = "aiperf.transports.base_transports"


def _patch(module) -> None:
    original = module.BaseTransport.build_headers

    def build_headers(self, request_info):
        headers = original(self, request_info)
        if request_info.x_correlation_id and request_info.is_final_turn:
            headers["X-Session-Final-Turn"] = "1"
        return headers

    module.BaseTransport.build_headers = build_headers


class _Loader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


sys.meta_path.insert(0, _Finder())
