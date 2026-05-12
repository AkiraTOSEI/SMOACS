from typing import Literal

import torch


def make_crystal_index(size: torch.Tensor) -> torch.Tensor:
    if size.ndim != 1:
        raise ValueError(f"size must be 1D, got shape {tuple(size.shape)}")
    if (size < 0).any():
        raise ValueError("size must be non-negative")
    size_long = size.to(dtype=torch.long)
    num_crystals = int(size_long.numel())
    return torch.repeat_interleave(
        torch.arange(num_crystals, device=size_long.device), size_long
    )


def segment_sum(
    values: torch.Tensor, idx: torch.Tensor, n_segments: int
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {tuple(values.shape)}")
    if idx.ndim != 1:
        raise ValueError(f"idx must be 1D, got shape {tuple(idx.shape)}")
    if values.shape[0] != idx.shape[0]:
        raise ValueError(
            f"values and idx must have same length, got {values.shape[0]} and {idx.shape[0]}"
        )
    if idx.dtype != torch.long:
        idx = idx.to(dtype=torch.long)
    out = torch.zeros(
        n_segments,
        values.shape[1],
        dtype=values.dtype,
        device=values.device,
    )
    return out.index_add(0, idx, values)


def element_count_l1_loss(
    atomic_distribution: torch.Tensor,
    size: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    if atomic_distribution.ndim != 2 or atomic_distribution.shape[1] != 98:
        raise ValueError(
            "atomic_distribution must have shape (N, 98), "
            f"got {tuple(atomic_distribution.shape)}"
        )
    if size.ndim != 1:
        raise ValueError(f"size must be 1D, got shape {tuple(size.shape)}")
    if (size < 0).any():
        raise ValueError("size must be non-negative")
    n_atoms = atomic_distribution.shape[0]
    if int(size.sum().item()) != n_atoms:
        raise ValueError(
            f"sum(size) must equal N={n_atoms}, got {int(size.sum().item())}"
        )

    idx = make_crystal_index(size)
    per_cry_mass = segment_sum(atomic_distribution, idx, int(size.numel()))
    per_crystal_loss = per_cry_mass.abs().sum(dim=1)

    if reduction == "none":
        return per_crystal_loss
    if reduction == "sum":
        return per_crystal_loss.sum()
    if reduction == "mean":
        return per_crystal_loss.mean()
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
