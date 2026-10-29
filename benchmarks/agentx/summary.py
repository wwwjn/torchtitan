"""Side-by-side table of run_agentx.sh's harness results (AIPerf profile_export_aiperf.json).

    python summary.py <out_dir> [<out_dir> ...]

Each <out_dir> holds one AIPerf artifact directory per system (titan, vllm). TitanRL's server streams
the first token, then the rest in one chunk: its TTFT, mean ITL and throughput are valid, its
per-chunk latencies (time to second token, inter-chunk) are not.
"""

import json
import os
import sys


def get(d, key, stat="avg"):
    value = d.get(key, {})
    return value.get(stat) if isinstance(value, dict) else None


header = ("run", "system", "out tok/s", "titan / vllm", "reqs", "ISL", "OSL", "TTFT p50 ms",
          "ITL avg ms", "ITL p50 ms", "cache %", "errors", "dur s")
print("| " + " | ".join(header) + " |")
print("|" + "---|" * len(header))
for out in sys.argv[1:]:
    results = {}
    for system in ("titan", "vllm"):
        path = os.path.join(out, system, "profile_export_aiperf.json")
        if os.path.exists(path):
            results[system] = json.load(open(path))
    ratio = ""
    if len(results) == 2:
        ratio = (f"{get(results['titan'], 'output_token_throughput') / get(results['vllm'], 'output_token_throughput'):.3f}")
    for system, d in results.items():
        print("| " + " | ".join((
            os.path.basename(os.path.normpath(out)), system,
            f"{get(d, 'output_token_throughput'):.0f}",
            ratio if system == "titan" else "",
            f"{get(d, 'request_count'):.0f}",
            f"{get(d, 'input_sequence_length'):,.0f}",
            f"{get(d, 'output_sequence_length'):.0f}",
            f"{get(d, 'time_to_first_token', 'p50'):.0f}",
            f"{get(d, 'inter_token_latency'):.1f}",
            f"{get(d, 'inter_token_latency', 'p50'):.1f}",
            f"{get(d, 'overall_usage_prompt_cache_read_pct'):.1f}",
            str(len(d.get("error_summary") or [])),
            f"{get(d, 'benchmark_duration'):.0f}",
        )) + " |")
