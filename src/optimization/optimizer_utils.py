from typing import Optional, Tuple

import torch


def define_optimizer_and_scheduler(
    lattice_lr: float,
    atom_lr: float,
    coords_lr: float,
    ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor],
    atomic_distribution: torch.Tensor,
    scaled_batch_abc: torch.Tensor,
    scaled_batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    angle_optimization: bool,
    num_steps: int,
    lattice_cycle: int,
    atom_cycle: int,
    coords_cycle: int,
):
    """
    Define optimizers and learning rate schedulers for lattice, atomic, and coordinate parameters.

    Args:
        lattice_lr (float): Learning rate for lattice parameters.
        atom_lr (float): Learning rate for atomic distribution and oxidation mask.
        coords_lr (float): Learning rate for atomic coordinates.
        ox_mask_learnable_tensor_per_crystal (Optional[torch.Tensor]): Learnable oxidation mask tensor.
        atomic_distribution (torch.Tensor): Atomic distribution tensor.
        scaled_batch_abc (torch.Tensor): Scaled lattice lengths.
        scaled_batch_angle (torch.Tensor): Scaled lattice angles.
        batch_dir_coords (torch.Tensor): Atomic coordinates (direct space).
        angle_optimization (bool): Whether to optimize lattice angles.
        num_steps (int): Total optimization steps.
        lattice_cycle (int): Scheduler period for lattice.
        atom_cycle (int): Scheduler period for atoms.
        coords_cycle (int): Scheduler period for coordinates.

    Returns:
        Tuple[torch.optim.Optimizer, ...]:
            Optimizers and schedulers for each component (lattice, atom, coords).
    """
    # optimizer and scheduler
    if ox_mask_learnable_tensor_per_crystal is not None:
        print("<info> atomic_mask is used")
        optimizer_atom = torch.optim.Adam(
            [atomic_distribution, ox_mask_learnable_tensor_per_crystal], lr=atom_lr
        )
    else:
        optimizer_atom = torch.optim.Adam([atomic_distribution], lr=atom_lr)

    if angle_optimization:
        print("<info> angle optimization is enabled")
        optimizer_lattice = torch.optim.Adam(
            [scaled_batch_abc, scaled_batch_angle], lr=lattice_lr
        )
    else:
        print("<info> angle optimization is disabled")
        optimizer_lattice = torch.optim.Adam([scaled_batch_abc], lr=lattice_lr)

    optimizer_coords = torch.optim.Adam([batch_dir_coords], lr=coords_lr)
    scheduler_lattice = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_lattice, T_max=num_steps // lattice_cycle, eta_min=0
    )
    scheduler_atom = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_atom, T_max=num_steps // atom_cycle, eta_min=0
    )
    scheduler_coords = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_coords, T_max=num_steps // coords_cycle, eta_min=0
    )

    return (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    )
