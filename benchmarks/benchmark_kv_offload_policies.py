# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Micro benchmark for CPU KV offloading eviction policies.

This benchmark exercises CPUOffloadingManager directly on synthetic access
traces, comparing cache hit rate and hot-block retention for:
    - lru
    - lfu
    - arc

Example:
    PYTHONPATH=$PWD python benchmarks/benchmark_kv_offload_policies.py
    PYTHONPATH=$PWD python benchmarks/benchmark_kv_offload_policies.py \
        --capacity 16 --trace frequency_skew --repetitions 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from vllm.v1.kv_offload.abstract import make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

PolicyName = Literal["lru", "lfu", "arc"]
TraceName = Literal["frequency_skew", "recency_skew", "mixed"]
POLICIES: tuple[PolicyName, ...] = ("lru", "lfu", "arc")
TRACES: tuple[TraceName, ...] = ("frequency_skew", "recency_skew", "mixed")


def _to_key(token_id: int) -> bytes:
    return make_offload_key(str(token_id).encode(), 0)


@dataclass
class BenchmarkResult:
    policy: PolicyName
    trace: TraceName
    accesses: int
    hits: int
    misses: int
    hit_rate: float
    hot_resident: bool


def _build_trace(
    trace: TraceName,
    capacity: int,
    repetitions: int,
) -> tuple[list[int], int]:
    hot_key = 1

    if trace == "frequency_skew":
        train = [hot_key] * repetitions
        refresh_cold = list(range(2, capacity + 1))
        pressure = list(range(capacity + 1, capacity + 1 + repetitions))
        return train + refresh_cold + pressure + [hot_key], hot_key

    if trace == "recency_skew":
        train = [hot_key] * repetitions
        stream = list(range(capacity + 1, capacity + 1 + capacity + repetitions))
        return train + stream + [hot_key], hot_key

    if trace == "mixed":
        train = [hot_key] * (repetitions // 2)
        mixed = []
        next_cold = capacity + 1
        for _ in range(repetitions):
            mixed.extend([hot_key, 2, next_cold])
            next_cold += 1
        return train + mixed + [hot_key], hot_key

    raise ValueError(f"Unsupported trace: {trace}")


def _run_trace(policy: PolicyName, trace: TraceName, capacity: int, repetitions: int) -> BenchmarkResult:
    manager = CPUOffloadingManager(
        block_size=256,
        num_blocks=capacity,
        cache_policy=policy,
        enable_events=False,
    )
    seed_keys = list(range(1, capacity + 1))
    seed_offload_keys = [_to_key(i) for i in seed_keys]
    manager.prepare_store(seed_offload_keys)
    manager.complete_store(seed_offload_keys)

    accesses, hot_key = _build_trace(trace, capacity, repetitions)

    hits = 0
    misses = 0
    for token_id in accesses:
        key = _to_key(token_id)
        if manager.lookup([key]) == 1:
            hits += 1
            manager.touch([key])
            continue

        misses += 1
        output = manager.prepare_store([key])
        if output is None:
            raise RuntimeError(
                f"Failed to store key {token_id} under policy={policy} trace={trace}"
            )
        manager.complete_store([key])

    hot_resident = manager.lookup([_to_key(hot_key)]) == 1
    total = hits + misses
    return BenchmarkResult(
        policy=policy,
        trace=trace,
        accesses=total,
        hits=hits,
        misses=misses,
        hit_rate=(hits / total) if total else 0.0,
        hot_resident=hot_resident,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capacity",
        type=int,
        default=8,
        help="Number of CPU offloaded blocks in the benchmark cache.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=8,
        help="Controls how strongly the traces emphasize hot-key reuse.",
    )
    parser.add_argument(
        "--trace",
        choices=TRACES,
        nargs="+",
        default=list(TRACES),
        help="Synthetic traces to run.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        nargs="+",
        default=list(POLICIES),
        help="Eviction policies to compare.",
    )
    args = parser.parse_args()

    print(
        f"Benchmarking KV offload policies with capacity={args.capacity}, "
        f"repetitions={args.repetitions}"
    )
    print(
        "trace            policy  accesses  hits  misses  hit_rate  hot_resident"
    )
    print(
        "---------------  ------  --------  ----  ------  --------  ------------"
    )

    for trace in args.trace:
        for policy in args.policy:
            result = _run_trace(policy, trace, args.capacity, args.repetitions)
            print(
                f"{result.trace:15s}  "
                f"{result.policy:6s}  "
                f"{result.accesses:8d}  "
                f"{result.hits:4d}  "
                f"{result.misses:6d}  "
                f"{result.hit_rate:8.3f}  "
                f"{str(result.hot_resident):>12s}"
            )


if __name__ == "__main__":
    main()
