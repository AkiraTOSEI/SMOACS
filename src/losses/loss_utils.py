from typing import Optional, Tuple

import torch


def calculate_loss_from_output(
    bandgap_pred: torch.Tensor,
    ef_pred: torch.Tensor,
    sharpened_ox_mask: torch.Tensor,
    normalized_dist: torch.Tensor,
    site_ids: torch.Tensor,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    ef_coef: float,
    e_form_min: Optional[float],
    crystal_system: Optional[str],
    radii_tensor: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Calculate total loss for bandgap, formation energy, and tolerance factor.

    This function combines bandgap loss, formation energy loss, and optionally tolerance loss (for perovskites),
    based on model outputs and auxiliary inputs like oxidation masks and site information.

    Args:
        bandgap_pred (torch.Tensor): Predicted bandgap values. Shape: (N,)
        ef_pred (torch.Tensor): Predicted formation energies. Shape: (N,)
        sharpened_ox_mask (torch.Tensor): Oxidation-state mask. Shape: (N, max_ox)
        normalized_dist (torch.Tensor): Atomic distribution after masking and softmax. Shape: (N, num_atoms)
        site_ids (torch.Tensor): Site type IDs (e.g., A/B/X). Shape: (N_atoms,)
        gap_loss_func (torch.nn.Module): Bandgap loss function.
        tolerance_loss_func (torch.nn.Module): Tolerance factor loss function.
        ef_coef (float): Weighting factor for formation energy loss.
        e_form_min (Optional[float]): Minimum target formation energy. If None, use raw ef_pred.
        crystal_system (Optional[str]): If "perovskite", enables tolerance loss computation.
        radii_tensor (torch.Tensor): Atomic radii tensor for tolerance factor computation. Shape: (N_atoms, num_atom_types, max_ox)

    Returns:
        Tuple[torch.Tensor, ...]:
            - total_loss: Scalar total loss.
            - total_each_loss: Elementwise sum of loss components.
            - gap_loss: Bandgap loss per sample.
            - ef_loss: Formation energy loss per sample.
            - tolerance_loss: Tolerance factor loss per sample (NaN if not perovskite).
            - tolerance: Tolerance values per structure (NaN if not perovskite).
    """
    gap_loss = gap_loss_func(bandgap_pred, reduction="none")
    if e_form_min is not None:
        ef_loss = torch.clip(ef_pred.squeeze() - e_form_min, min=0) * ef_coef
    else:
        ef_loss = ef_pred.squeeze() * ef_coef
    loss = gap_loss + ef_loss

    # tolerance loss
    if crystal_system == "perovskite":
        num_batch_crystal = len(gap_loss)  # number of crystals in a batch
        # calculate mean radii at each site normalized by the mean of oxides
        radii_per_atom = torch.sum(
            sharpened_ox_mask.unsqueeze(1) * radii_tensor, dim=2
        )  # shape=(N, num_type_of_atoms)
        site_mean_raddi = torch.sum(normalized_dist * radii_per_atom, dim=1).view(
            num_batch_crystal, -1
        )
        # calculate mean raddi at each crystal. A_site: 0, B_site: 1, X_site: 2
        A_site_mask = site_ids.view(num_batch_crystal, -1) == 0
        A_site_raddi = torch.where(
            A_site_mask,
            site_mean_raddi,
            torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device),
        ).sum(dim=1) / A_site_mask.sum(dim=1)
        B_site_mask = site_ids.view(num_batch_crystal, -1) == 1
        B_site_raddi = torch.where(
            B_site_mask,
            site_mean_raddi,
            torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device),
        ).sum(dim=1) / B_site_mask.sum(dim=1)
        X_site_mask = site_ids.view(num_batch_crystal, -1) == 2
        X_site_raddi = torch.where(
            X_site_mask,
            site_mean_raddi,
            torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device),
        ).sum(dim=1) / X_site_mask.sum(dim=1)
        # testing
        torch.testing.assert_close(
            A_site_mask.sum(dim=1), B_site_mask.sum(dim=1), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            A_site_mask.sum(dim=1) * 3, X_site_mask.sum(dim=1), atol=1e-5, rtol=1e-5
        )
        # calculate tolerance and its loss
        tolerance = (A_site_raddi + X_site_raddi) / (
            (2**0.5) * (B_site_raddi + X_site_raddi)
        )
        tolerance_loss = tolerance_loss_func(tolerance)
        loss = loss + tolerance_loss

    else:
        tolerance = torch.zeros_like(gap_loss).squeeze() * torch.nan
        tolerance_loss = torch.zeros_like(gap_loss).squeeze() * torch.nan

    total_each_loss = torch.nan_to_num(loss, nan=0.0)

    total_loss = total_each_loss.mean()
    return total_loss, total_each_loss, gap_loss, ef_loss, tolerance_loss, tolerance
