#!/usr/bin/env python3
from __future__ import annotations

"""
DP-based uniform sampler for ordered integer sequences with a target sum.

Example:
    states = [i for i in range(-4, 8) if i != 0]
    samples = sample_k(n=50, states=states, k=10, target=0, seed=0, unique=True)
    assert all(len(s) == 50 and sum(s) == 0 for s in samples)
"""

from typing import Iterable, List, Sequence, Tuple
import random


def build_ways(n: int, states: Sequence[int], target: int = 0) -> tuple[list[list[int]], int]:
    if n < 0:
        raise ValueError("n must be >= 0")
    if not states and n > 0:
        max_abs = 0
    else:
        max_abs = max((abs(v) for v in states), default=0)
    offset = n * max_abs
    width = 2 * offset + 1
    ways: list[list[int]] = [[0] * width for _ in range(n + 1)]
    ways[0][offset] = 1

    for rem in range(1, n + 1):
        prev = ways[rem - 1]
        cur = ways[rem]
        for idx in range(width):
            total = 0
            for v in states:
                prev_idx = idx - v
                if 0 <= prev_idx < width:
                    total += prev[prev_idx]
            cur[idx] = total

    return ways, offset


def build_ways_per_pos(
    states_per_pos: Sequence[Sequence[int]],
    target: int = 0,
) -> tuple[list[list[int]], int]:
    n = len(states_per_pos)
    if n == 0:
        offset = 0
        return [[1]], offset
    max_abs = 0
    for choices in states_per_pos:
        if not choices:
            return [[0]], 0
        max_abs = max(max_abs, max(abs(v) for v in choices))
    offset = n * max_abs
    width = 2 * offset + 1
    ways: list[list[int]] = [[0] * width for _ in range(n + 1)]
    ways[0][offset] = 1
    for i in range(1, n + 1):
        prev = ways[i - 1]
        cur = ways[i]
        for idx in range(width):
            total = 0
            for v in states_per_pos[i - 1]:
                prev_idx = idx - v
                if 0 <= prev_idx < width:
                    total += prev[prev_idx]
            cur[idx] = total
    return ways, offset


def count_solutions(n: int, states: Sequence[int], target: int = 0) -> int:
    ways, offset = build_ways(n, states, target=target)
    idx = target + offset
    if idx < 0 or idx >= len(ways[n]):
        return 0
    return ways[n][idx]


def sample_one(
    n: int,
    states: Sequence[int],
    target: int = 0,
    rng: random.Random | None = None,
) -> tuple[int, ...]:
    if rng is None:
        rng = random.Random()
    ways, offset = build_ways(n, states, target=target)
    idx = target + offset
    if idx < 0 or idx >= len(ways[n]) or ways[n][idx] == 0:
        raise ValueError("No solutions for the given parameters.")

    seq: list[int] = []
    need = target
    for rem in range(n, 0, -1):
        total = 0
        counts = []
        for v in states:
            prev_need = need - v
            prev_idx = prev_need + offset
            if 0 <= prev_idx < len(ways[rem - 1]):
                count = ways[rem - 1][prev_idx]
            else:
                count = 0
            counts.append(count)
            total += count
        if total == 0:
            raise ValueError("No solutions for the given parameters.")

        r = rng.randrange(total)
        cumulative = 0
        chosen = None
        for v, count in zip(states, counts):
            cumulative += count
            if r < cumulative:
                chosen = v
                break
        if chosen is None:
            raise RuntimeError("Sampling failed to choose a value.")
        seq.append(chosen)
        need -= chosen

    return tuple(seq)


def sample_one_per_pos(
    states_per_pos: Sequence[Sequence[int]],
    target: int = 0,
    rng: random.Random | None = None,
) -> tuple[int, ...]:
    if rng is None:
        rng = random.Random()
    ways, offset = build_ways_per_pos(states_per_pos, target=target)
    n = len(states_per_pos)
    idx = target + offset
    if idx < 0 or idx >= len(ways[n]) or ways[n][idx] == 0:
        raise ValueError("No solutions for the given parameters.")

    seq: list[int] = []
    need = target
    for i in range(n, 0, -1):
        choices = states_per_pos[i - 1]
        total = 0
        counts = []
        for v in choices:
            prev_need = need - v
            prev_idx = prev_need + offset
            if 0 <= prev_idx < len(ways[i - 1]):
                count = ways[i - 1][prev_idx]
            else:
                count = 0
            counts.append(count)
            total += count
        if total == 0:
            raise ValueError("No solutions for the given parameters.")
        r = rng.randrange(total)
        cumulative = 0
        chosen = None
        for v, count in zip(choices, counts):
            cumulative += count
            if r < cumulative:
                chosen = v
                break
        if chosen is None:
            raise RuntimeError("Sampling failed to choose a value.")
        seq.append(chosen)
        need -= chosen
    return tuple(seq)


def sample_k_per_pos(
    states_per_pos: Sequence[Sequence[int]],
    k: int = 10,
    target: int = 0,
    seed: int | None = None,
    unique: bool = True,
) -> list[tuple[int, ...]]:
    if k < 0:
        raise ValueError("k must be >= 0")
    ways, offset = build_ways_per_pos(states_per_pos, target=target)
    n = len(states_per_pos)
    idx = target + offset
    total = 0
    if 0 <= idx < len(ways[n]):
        total = ways[n][idx]
    if total == 0:
        raise ValueError("No solutions for the given parameters.")
    if unique and k > total:
        raise ValueError("Requested more unique samples than total solutions.")
    rng = random.Random(seed)
    if not unique:
        return [sample_one_per_pos(states_per_pos, target=target, rng=rng) for _ in range(k)]
    samples: list[tuple[int, ...]] = []
    seen = set()
    while len(samples) < k:
        seq = sample_one_per_pos(states_per_pos, target=target, rng=rng)
        if seq in seen:
            continue
        seen.add(seq)
        samples.append(seq)
    return samples


def sample_k(
    n: int,
    states: Sequence[int],
    k: int = 10,
    target: int = 0,
    seed: int | None = None,
    unique: bool = True,
) -> list[tuple[int, ...]]:
    if k < 0:
        raise ValueError("k must be >= 0")
    total = count_solutions(n, states, target=target)
    if total == 0:
        raise ValueError("No solutions for the given parameters.")
    if unique and k > total:
        raise ValueError("Requested more unique samples than total solutions.")

    rng = random.Random(seed)
    samples: list[tuple[int, ...]] = []
    if not unique:
        for _ in range(k):
            samples.append(sample_one(n, states, target=target, rng=rng))
        return samples

    seen = set()
    while len(samples) < k:
        seq = sample_one(n, states, target=target, rng=rng)
        if seq in seen:
            continue
        seen.add(seq)
        samples.append(seq)
    return samples


__all__ = [
    "build_ways",
    "build_ways_per_pos",
    "count_solutions",
    "sample_one",
    "sample_one_per_pos",
    "sample_k",
    "sample_k_per_pos",
]
