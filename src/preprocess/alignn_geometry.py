"""Differentiable geometry shared by non-Wyckoff ALIGNN graph and forward paths."""

from typing import Optional

import torch

from src.utils.coord import compute_lattice_vectors


def prepare_alignn_geometry(
    scaled_batch_abc: torch.Tensor,
    scaled_batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    *,
    abc_scaler,
    angle_scaler,
    init_coords: torch.Tensor,
    limit_coords_displacement: Optional[float],
):
    """Return abc, angles, lattice, unwrapped geometry, and wrapped save coordinates.

    Keep the unwrapped representation for both graph images and differentiable
    edge vectors. Only the graph builder may detach these outputs. Noise, if
    requested by the caller, must already have been added once to the inputs.
    """
    batch_abc = abc_scaler.rescale(scaled_batch_abc)
    batch_angle = angle_scaler.rescale(scaled_batch_angle)
    coords_geometry = batch_dir_coords
    if limit_coords_displacement is not None:
        coords_geometry = torch.clamp(
            coords_geometry,
            min=init_coords - limit_coords_displacement,
            max=init_coords + limit_coords_displacement,
        )
    return (
        batch_abc,
        batch_angle,
        compute_lattice_vectors(batch_abc, batch_angle),
        coords_geometry,
        torch.remainder(coords_geometry, 1.0),
    )
