#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
from jarvis.core.atoms import pmg_to_atoms
from pymatgen.core import Lattice, Structure
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LatticeRandomizeResult:
    structure: Structure
    rho_target: float
    volume_target: float
    iso_scale: float
    u1: float
    u2: float
    min_dist: float
    rho_actual: float
    retry_counts: dict[str, int]

    def to_metadata(self) -> dict:
        return {
            "rho_target": self.rho_target,
            "volume_target": self.volume_target,
            "iso_scale": self.iso_scale,
            "u1": self.u1,
            "u2": self.u2,
            "min_dist": self.min_dist,
            "rho_actual": self.rho_actual,
            "retry_counts": self.retry_counts,
        }


def _shape_matrix(u1: float, u2: float) -> np.ndarray:
    u3 = -u1 - u2
    s1, s2, s3 = np.exp([u1, u2, u3])
    return np.diag([s1, s2, s3])


def check_nearest_neighbors_distance(
    structure: Structure,
    r: float = 1.0,
    verbose: int = 0,
    return_shortest_dist: bool = False,
) -> float | bool:
    if r <= 0:
        raise ValueError("r must be positive.")
    atoms = pmg_to_atoms(structure)
    supercell_atoms = atoms.make_supercell([2, 2, 2])
    coords = np.array(supercell_atoms.cart_coords)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r)
    if not pairs:
        return float("inf") if return_shortest_dist else True
    if return_shortest_dist:
        min_dist = min(np.linalg.norm(coords[i] - coords[j]) for i, j in pairs)
        return float(min_dist)
    if verbose:
        for i, j in pairs:
            print(f"close pair: {i}, {j}")
    return False


def randomize_lattice_with_density(
    structure: Structure,
    rho_min: float,
    rho_max: float,
    min_dist_threshold: float = 1.0,
    rng: np.random.Generator | None = None,
    max_density_trials: int = 30,
    max_shape_trials: int = 50,
    shape_scale: float = 0.4,
    profile: dict[str, float] | None = None,
    verbose: bool = False,
    log_every_density: int = 5,
    log_every_shape: int = 20,
) -> LatticeRandomizeResult:
    if rng is None:
        rng = np.random.default_rng()
    if rho_min <= 0 or rho_max <= 0:
        raise ValueError("rho_min and rho_max must be positive.")
    if rho_min > rho_max:
        raise ValueError("rho_min must be <= rho_max.")

    nsites = structure.num_sites
    base_volume = structure.volume
    retry_counts = {"density": 0, "shape": 0}

    if profile is not None:
        profile.setdefault("density_trials", 0)
        profile.setdefault("shape_trials", 0)
        profile.setdefault("min_dist_time", 0.0)
        profile.setdefault("min_dist_fail", 0)
        profile.setdefault("rho_fail", 0)

    for density_i in range(max_density_trials):
        if verbose and density_i % log_every_density == 0:
            print(
                f"[randomize] density trial {density_i+1}/{max_density_trials}",
                flush=True,
            )
        if profile is not None:
            profile["density_trials"] += 1
        rho_target = float(rng.uniform(rho_min, rho_max))
        volume_target = float(nsites / rho_target)
        iso_scale = float((volume_target / base_volume) ** (1.0 / 3.0))

        scaled = structure.copy()
        scaled.scale_lattice(volume_target)

        for shape_i in range(max_shape_trials):
            if verbose and shape_i % log_every_shape == 0:
                print(
                    f"[randomize] shape trial {shape_i+1}/{max_shape_trials}",
                    flush=True,
                )
            if profile is not None:
                profile["shape_trials"] += 1
            u1 = float(rng.uniform(-shape_scale, shape_scale))
            u2 = float(rng.uniform(-shape_scale, shape_scale))
            shape_mat = _shape_matrix(u1, u2)
            new_matrix = scaled.lattice.matrix @ shape_mat
            new_lattice = Lattice(new_matrix)
            new_structure = Structure(new_lattice, scaled.species, scaled.frac_coords)

            t0 = time.perf_counter()
            min_dist = float(
                check_nearest_neighbors_distance(
                    new_structure, r=min_dist_threshold, return_shortest_dist=True
                )
            )
            if profile is not None:
                profile["min_dist_time"] += time.perf_counter() - t0
            rho_actual = float(new_structure.num_sites / new_structure.volume)

            if min_dist < min_dist_threshold:
                if profile is not None:
                    profile["min_dist_fail"] += 1
                continue
            if not (rho_min <= rho_actual <= rho_max):
                if profile is not None:
                    profile["rho_fail"] += 1
                continue
            if min_dist >= min_dist_threshold and rho_min <= rho_actual <= rho_max:
                retry_counts["density"] = density_i
                retry_counts["shape"] = shape_i
                return LatticeRandomizeResult(
                    structure=new_structure,
                    rho_target=rho_target,
                    volume_target=volume_target,
                    iso_scale=iso_scale,
                    u1=u1,
                    u2=u2,
                    min_dist=min_dist,
                    rho_actual=rho_actual,
                    retry_counts=retry_counts,
                )

        retry_counts["shape"] += max_shape_trials

    raise RuntimeError("Failed to randomize lattice within retry limits.")


__all__ = [
    "LatticeRandomizeResult",
    "randomize_lattice_with_density",
]
