#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from scripts.data_generations.oxidation_sampler import build_ways

_CACHE: dict[str, dict] = {}


def _sample_one_from_ways(
    ways: list[list[int]],
    offset: int,
    states: Sequence[int],
    target: int,
    rng: random.Random,
) -> tuple[int, ...]:
    n = len(ways) - 1
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


def build_neutral_ox_dict(
    states: Sequence[int],
    n_max: int = 135,
    k_per_n: int = 1000,
    target: int = 0,
    seed: int | None = None,
) -> dict:
    rng = random.Random(seed)
    data: dict[str, list[list[int]]] = {}
    max_abs = max((abs(v) for v in states), default=0)
    for n in range(1, n_max + 1):
        if target < -n * max_abs or target > n * max_abs:
            data[str(n)] = []
            continue
        ways, offset = build_ways(n, states, target=target)
        total = 0
        idx = target + offset
        if 0 <= idx < len(ways[n]):
            total = ways[n][idx]
        if total == 0:
            data[str(n)] = []
            continue
        k_eff = min(k_per_n, total)
        samples: list[list[int]] = []
        seen = set()
        while len(samples) < k_eff:
            seq = _sample_one_from_ways(ways, offset, states, target, rng)
            if seq in seen:
                continue
            seen.add(seq)
            samples.append(list(seq))
        data[str(n)] = samples
    return data


def save_neutral_ox_dict(
    out_path: Path,
    states: Sequence[int],
    n_max: int = 135,
    k_per_n: int = 1000,
    target: int = 0,
    seed: int | None = None,
) -> dict:
    payload = {
        "meta": {
            "n_max": n_max,
            "k_per_n": k_per_n,
            "target": target,
        },
        "data": build_neutral_ox_dict(
            states=states,
            n_max=n_max,
            k_per_n=k_per_n,
            target=target,
            seed=seed,
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    _CACHE[str(out_path)] = payload
    return payload


def load_or_build_neutral_ox_dict(
    out_dir: Path,
    mask_name: str,
    states: Sequence[int],
    n_max: int = 135,
    k_per_n: int = 1000,
    target: int = 0,
    seed: int | None = None,
    overwrite: bool = True,
) -> dict:
    out_path = out_dir / f"neutral_ox_by_n__{mask_name}.json"
    cached = _CACHE.get(str(out_path))
    if cached is not None:
        return cached
    if out_path.exists() and not overwrite:
        payload = json.loads(out_path.read_text())
        _CACHE[str(out_path)] = payload
        return payload
    return save_neutral_ox_dict(
        out_path=out_path,
        states=states,
        n_max=n_max,
        k_per_n=k_per_n,
        target=target,
        seed=seed,
    )


__all__ = [
    "build_neutral_ox_dict",
    "save_neutral_ox_dict",
    "load_or_build_neutral_ox_dict",
]
