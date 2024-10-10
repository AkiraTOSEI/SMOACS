from typing import Optional

import torch


def define_optimizer_and_scheduler(
    lattice_lr:float,
    atom_lr:float,
    coords_lr:float,
    ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor],
    atomic_distribution: torch.Tensor,
    scaled_batch_abc: torch.Tensor,
    scaled_batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    angle_optimization:bool,
    length_optimization:bool,
    num_steps:int,
    lattice_cycle:int,
    atom_cycle:int,
    coords_cycle:int,
):
    # optimizer and scheduler
    if ox_mask_learnable_tensor_per_crystal is not None:
        print('<info> atomic_mask is used')
        optimizer_atom = torch.optim.Adam([atomic_distribution, ox_mask_learnable_tensor_per_crystal], lr=atom_lr)
    else:
        optimizer_atom = torch.optim.Adam([atomic_distribution], lr=atom_lr)
        
    if angle_optimization and length_optimization:
        print('<info> angle and length optimization are enabled')
        optimizer_lattice = torch.optim.Adam([scaled_batch_abc, scaled_batch_angle], lr=lattice_lr)
        
    elif angle_optimization and (not length_optimization):
        print('<info> only angle optimization is enabled')
        optimizer_lattice = torch.optim.Adam([scaled_batch_angle], lr=lattice_lr)
    elif (not angle_optimization) and length_optimization:
        print('<info> angle optimization is disabled')
        optimizer_lattice = torch.optim.Adam([scaled_batch_abc], lr=lattice_lr)
    else:
        raise ValueError('angle_optimization and length_optimization cannot be both False')
    
    optimizer_coords = torch.optim.Adam([batch_dir_coords], lr=coords_lr)
    scheduler_lattice = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_lattice, T_max=num_steps//lattice_cycle, eta_min=0)
    scheduler_atom = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_atom, T_max=num_steps//atom_cycle, eta_min=0)
    scheduler_coords = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_coords, T_max=num_steps//coords_cycle, eta_min=0)

    return optimizer_lattice, optimizer_atom, optimizer_coords, scheduler_lattice, scheduler_atom, scheduler_coords