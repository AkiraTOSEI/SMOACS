"""Structure-weighted force and hydrostatic pressure losses (eV/Å and GPa)."""

import torch


DIAGNOSTIC_KEYS = (
    "force_loss_raw",
    "force_loss_transformed",
    "force_loss_weighted",
    "pressure_loss_raw",
    "pressure_loss_transformed",
    "pressure_loss_weighted",
    "max_force_eV_per_A",
    "pressure_GPa",
    "deviatoric_stress_norm_GPa",
)
FINAL_KEYS = tuple(
    f"{key}_{state}" for key in DIAGNOSTIC_KEYS for state in ("soft", "onehot")
) + ("loss_total_soft", "loss_total_onehot")


def active_force_pressure(config):
    if config is None:
        return False, False
    return (
        bool(config["use_force_loss"] and config["F_ref"] > 0),
        bool(config["use_pressure_loss"] and config["P_ref"] > 0),
    )


def force_pressure_losses(properties, sizes, config, structure_ids=None):
    """Return per-structure contributions, retaining the autograd graph."""
    force_on, pressure_on = active_force_pressure(config)
    result = {}
    if (force_on or pressure_on) and (sizes.ndim != 1 or (sizes <= 0).any()):
        raise ValueError("sizes must contain a positive atom count for every structure")
    if force_on:
        forces = properties["forces"]
        if forces.shape != (int(sizes.sum()), 3):
            raise ValueError("forces shape does not match sizes")
        per_atom = forces.square().sum(dim=-1)
        chunks = torch.split(per_atom, sizes.detach().cpu().tolist())
        result["force_loss_raw"] = torch.stack([chunk.mean() for chunk in chunks])
        result["max_force_eV_per_A"] = torch.stack(
            [chunk.max().sqrt() for chunk in chunks]
        )
    if pressure_on:
        stress = properties["stresses"]
        if stress.shape != (len(sizes), 3, 3):
            raise ValueError("stresses shape does not match sizes")
        transform_loss(stress.abs().amax(dim=(-2, -1)), "raw", 1.0, structure_ids)
        pressure = -stress.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
        result["pressure_GPa"] = pressure
        result["pressure_loss_raw"] = pressure.square()
        deviatoric = stress + pressure[:, None, None] * torch.eye(
            3, device=stress.device, dtype=stress.dtype
        )
        result["deviatoric_stress_norm_GPa"] = torch.linalg.matrix_norm(deviatoric)
    for term, on, coefficient in (
        ("force", force_on, "F_ref"),
        ("pressure", pressure_on, "P_ref"),
    ):
        if not on:
            continue
        raw = result[f"{term}_loss_raw"]
        transformed = transform_loss(
            raw, config[f"{term}_loss_mode"], config[f"{term}_loss_clip"], structure_ids
        )
        result[f"{term}_loss_transformed"] = transformed
        result[f"{term}_loss_weighted"] = config[coefficient] * transformed
    return result


def transform_loss(raw, mode, clip, structure_ids=None):
    invalid = ~torch.isfinite(raw) | (raw < 0)
    if invalid.any():
        indices = invalid.nonzero().flatten().tolist()
        ids = (
            indices
            if structure_ids is None
            else [str(structure_ids[i]) for i in indices]
        )
        raise ValueError(f"Nonfinite or negative raw loss for structures {ids}")
    if mode == "log":
        return raw.clamp_min(clip).log10()
    if mode == "raw":
        return raw
    raise ValueError(f"Unknown force/pressure loss mode: {mode}")


def weighted_force_pressure(diagnostics):
    return sum(
        value for key, value in diagnostics.items() if key.endswith("_loss_weighted")
    )


def final_force_pressure(
    module,
    config,
    soft,
    onehot,
    positions,
    lattice,
    sizes,
    base_soft,
    base_onehot,
    energy_coef=0.0,
    structure_ids=None,
):
    """Evaluate both final distributions with exactly one forward for each state."""
    force_on, pressure_on = active_force_pressure(config)
    result = {}
    for state, distribution, base in (
        ("soft", soft, base_soft),
        ("onehot", onehot, base_onehot),
    ):
        properties = module.compute_properties(
            atom_types=distribution,
            positions=positions,
            lattice=lattice,
            sizes=sizes,
            include_forces=force_on,
            include_stresses=pressure_on,
            create_graph=False,
        )
        diagnostics = force_pressure_losses(properties, sizes, config, structure_ids)
        result.update(
            {f"{key}_{state}": value.detach() for key, value in diagnostics.items()}
        )
        total = base.reshape(-1) + weighted_force_pressure(diagnostics)
        if energy_coef > 0:
            energy_per_atom = properties["total_energy"] / sizes.to(
                properties["total_energy"].dtype
            )
            total = total + energy_coef * energy_per_atom
            if state == "onehot":
                result["pred_energy"] = energy_per_atom.detach()
        result[f"loss_total_{state}"] = total.detach()
    result["loss_total"] = result["loss_total_onehot"]
    return result
