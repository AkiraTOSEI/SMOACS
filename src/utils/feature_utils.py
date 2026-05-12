from typing import Dict, Optional, Tuple

import torch


def atom_dist_to_features(atomic_distribution, atom_feat_matrix):
    """
    Convert atomic distribution to atomic features.
    """
    atomic_features = torch.matmul(atomic_distribution, atom_feat_matrix)
    return atomic_features


def _build_onehot(indices: torch.Tensor, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build a row-wise one-hot tensor from index labels.

    Args:
        indices: Row-wise class indices with shape ``(N,)``.
        shape: Target one-hot shape ``(N, C)``.
        device: Device for the output tensor.
        dtype: Data type for the output tensor.

    Returns:
        One-hot encoded tensor with shape ``shape``.
    """
    onehot = torch.zeros(shape, device=device, dtype=dtype)
    onehot[torch.arange(shape[0], device=device), indices.to(device)] = 1
    return onehot


def _expand_wp_tensor_to_atom_axis(
    wp_tensor: torch.Tensor,
    expected_rows: int,
    output_dict: Dict[str, torch.Tensor],
    tensor_name: str,
    target_device: torch.device,
) -> torch.Tensor:
    """Expand a WP-axis tensor to atom-axis using the Wyckoff expander.

    If the input already matches the expected atom-axis row count, this function
    only moves it to ``target_device``.

    Args:
        wp_tensor: Tensor on Wyckoff-position axis with shape ``(N_wp, F)``.
        expected_rows: Expected atom-axis row count ``N_atom``.
        output_dict: Dictionary that may include ``wyckoff_expander`` and ``site_ids``.
        tensor_name: Name used in assertion error messages.
        target_device: Device for the returned tensor.

    Returns:
        Tensor on atom axis with shape ``(N_atom, F)``.

    Raises:
        AssertionError: If expansion is required but necessary objects are missing,
            or if the expanded row count does not match ``expected_rows``.
    """
    if wp_tensor.shape[0] == expected_rows:
        return wp_tensor.to(target_device)

    wyckoff_expander = output_dict.get("wyckoff_expander")
    site_ids = output_dict.get("site_ids")
    assert wyckoff_expander is not None, (
        f"wyckoff_expander is required to expand {tensor_name}: "
        f"wp_rows={wp_tensor.shape[0]}, atom_rows={expected_rows}"
    )
    assert site_ids is not None, f"site_ids is required to expand {tensor_name}"

    expand_device = torch.device(wyckoff_expander.device)
    wp_tensor = wp_tensor.to(expand_device)
    site_ids = site_ids.to(expand_device)
    dummy_wp_coords = torch.zeros(
        (wp_tensor.shape[0], 3),
        device=expand_device,
        dtype=wp_tensor.dtype,
    )
    _, expanded_tensor = wyckoff_expander.wyckoff_expand(
        wp_coords=dummy_wp_coords,
        wp_atom_dist=wp_tensor,
        site_ids=site_ids,
    )
    assert expanded_tensor.shape[0] == expected_rows, (
        f"Expanded {tensor_name} row mismatch: "
        f"{expanded_tensor.shape[0]} vs expected {expected_rows}"
    )
    return expanded_tensor.to(target_device)


def calculate_onehot(
    output_dict: Dict[str, torch.Tensor],
    atom_feat_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode atom and oxidation predictions into one-hot tensors.

    Decoding rule:
      1) ``k* = argmax(sharpened_ox_mask)`` on WP axis.
      2) Select allowed atom mask for ``k*`` from ``atomic_mask``.
      3) ``x* = argmax(atomic_distribution * selected_atom_mask)`` on WP axis.
      4) Expand WP-axis one-hot tensors to atom axis when needed.

    Args:
        output_dict: Dictionary containing:
            ``normalized_dist``, ``sharpened_ox_mask``,
            ``atomic_distribution``, ``atomic_mask`` and, if expansion is needed,
            ``site_ids`` and ``wyckoff_expander``.
        atom_feat_matrix: Optional atom feature matrix with shape ``(num_types, feat_dim)``.

    Returns:
        A tuple ``(max_val, onehot_x, onehot_atom_feat, onehot_ox_mask)`` where:
            - ``max_val``: Maximum values from ``normalized_dist`` per atom ``(N_atom,)``.
            - ``onehot_x``: Decoded atom one-hot tensor ``(N_atom, num_types)``.
            - ``onehot_atom_feat``: Atom features from ``onehot_x`` ``(N_atom, feat_dim)``
              or NaN tensor when ``atom_feat_matrix`` is ``None``.
            - ``onehot_ox_mask``: Decoded oxidation one-hot tensor ``(N_atom, max_ox)``.

    Raises:
        AssertionError: If required keys/shapes are missing or inconsistent, or if
            decoded labels violate the selected mask constraints.
    """
    required_keys = {
        "normalized_dist",
        "sharpened_ox_mask",
        "atomic_distribution",
        "atomic_mask",
    }
    assert required_keys.issubset(output_dict), (
        f"calculate_onehot requires keys: {required_keys}, missing: {required_keys - set(output_dict.keys())}"
    )

    normalized_dist = output_dict["normalized_dist"]
    sharpened_ox_mask = output_dict["sharpened_ox_mask"]
    atomic_distribution = output_dict["atomic_distribution"]
    atomic_mask = output_dict["atomic_mask"]

    assert normalized_dist.ndim == 2, f"normalized_dist.ndim must be 2, got {normalized_dist.ndim}"
    assert sharpened_ox_mask.ndim == 2, f"sharpened_ox_mask.ndim must be 2, got {sharpened_ox_mask.ndim}"
    assert atomic_distribution.ndim == 2, f"atomic_distribution.ndim must be 2, got {atomic_distribution.ndim}"
    assert atomic_mask.ndim == 3, f"atomic_mask.ndim must be 3, got {atomic_mask.ndim}"
    assert atomic_distribution.shape[0] == atomic_mask.shape[0] == sharpened_ox_mask.shape[0], (
        "WP row mismatch among atomic_distribution / atomic_mask / sharpened_ox_mask: "
        f"{atomic_distribution.shape}, {atomic_mask.shape}, {sharpened_ox_mask.shape}"
    )
    assert atomic_distribution.shape[1] == atomic_mask.shape[1], (
        f"Atom-type dim mismatch: {atomic_distribution.shape} vs {atomic_mask.shape}"
    )
    assert atomic_mask.shape[2] == sharpened_ox_mask.shape[1], (
        f"Ox-state dim mismatch: {atomic_mask.shape} vs {sharpened_ox_mask.shape}"
    )

    max_val, _ = torch.max(normalized_dist, dim=1)
    _, max_ox_index_wp = torch.max(sharpened_ox_mask, dim=1)
    onehot_ox_wp = _build_onehot(
        indices=max_ox_index_wp,
        shape=sharpened_ox_mask.shape,
        device=sharpened_ox_mask.device,
        dtype=sharpened_ox_mask.dtype,
    )
    selected_atom_mask = torch.sum(
        atomic_mask * onehot_ox_wp.unsqueeze(1).to(atomic_mask.dtype),
        dim=2,
    )
    assert selected_atom_mask.shape == atomic_distribution.shape
    assert (selected_atom_mask.sum(dim=1) > 0).all(), "No allowed atom exists for selected k* in some WP rows"

    wp_scores = torch.clamp(atomic_distribution, min=1e-6) * selected_atom_mask
    _, max_index_wp = torch.max(wp_scores, dim=1)
    row_ids = torch.arange(selected_atom_mask.shape[0], device=selected_atom_mask.device)
    decoded_allowed = selected_atom_mask[row_ids, max_index_wp] > 0
    if not decoded_allowed.all():
        bad_rows = (~decoded_allowed).nonzero(as_tuple=True)[0]
        max_show = 8
        show_rows = bad_rows[:max_show].tolist()

        atomic_dist_isfinite = torch.isfinite(atomic_distribution)
        wp_scores_isfinite = torch.isfinite(wp_scores)
        ox_mask_isfinite = torch.isfinite(sharpened_ox_mask)

        row_summaries = []
        for row in show_rows:
            pred_x = int(max_index_wp[row].item())
            pred_k = int(max_ox_index_wp[row].item())
            row_summaries.append(
                "row="
                f"{row},k*={pred_k},x*={pred_x},allowed_at_x*="
                f"{float(selected_atom_mask[row, pred_x].item()):.1f},"
                "selected_row_sum="
                f"{float(selected_atom_mask[row].sum().item()):.1f},"
                "atomic_nan="
                f"{int(torch.isnan(atomic_distribution[row]).sum().item())},"
                "atomic_inf="
                f"{int(torch.isinf(atomic_distribution[row]).sum().item())},"
                "score_nan="
                f"{int(torch.isnan(wp_scores[row]).sum().item())},"
                "score_inf="
                f"{int(torch.isinf(wp_scores[row]).sum().item())},"
                "finite_atomic="
                f"{int(atomic_dist_isfinite[row].sum().item())}/{atomic_distribution.shape[1]},"
                "finite_score="
                f"{int(wp_scores_isfinite[row].sum().item())}/{wp_scores.shape[1]}"
            )

        raise AssertionError(
            "Decoded x* violates selected k* atomic mask: "
            f"bad_rows={int(bad_rows.numel())}; "
            "nonfinite_counts("
            f"atomic_distribution={int((~atomic_dist_isfinite).sum().item())}, "
            f"wp_scores={int((~wp_scores_isfinite).sum().item())}, "
            f"sharpened_ox_mask={int((~ox_mask_isfinite).sum().item())}"
            "); "
            "sample_rows=[" + "; ".join(row_summaries) + "]"
        )
    onehot_x_wp = _build_onehot(
        indices=max_index_wp,
        shape=atomic_distribution.shape,
        device=atomic_distribution.device,
        dtype=atomic_distribution.dtype,
    )

    expected_rows = normalized_dist.shape[0]
    onehot_x = _expand_wp_tensor_to_atom_axis(
        wp_tensor=onehot_x_wp,
        expected_rows=expected_rows,
        output_dict=output_dict,
        tensor_name="onehot_x",
        target_device=normalized_dist.device,
    )
    onehot_ox_mask = _expand_wp_tensor_to_atom_axis(
        wp_tensor=onehot_ox_wp,
        expected_rows=expected_rows,
        output_dict=output_dict,
        tensor_name="onehot_ox_mask",
        target_device=normalized_dist.device,
    )

    assert torch.allclose(
        onehot_x.sum(dim=1),
        torch.ones(onehot_x.shape[0], device=onehot_x.device, dtype=onehot_x.dtype),
    )
    assert torch.allclose(
        onehot_ox_mask.sum(dim=1),
        torch.ones(
            onehot_ox_mask.shape[0],
            device=onehot_ox_mask.device,
            dtype=onehot_ox_mask.dtype,
        ),
    )

    if atom_feat_matrix is not None:
        onehot_atom_feat = torch.matmul(onehot_x, atom_feat_matrix.to(onehot_x.device))
    else:
        onehot_atom_feat = torch.zeros_like(onehot_x) * torch.nan
    return max_val, onehot_x, onehot_atom_feat, onehot_ox_mask
