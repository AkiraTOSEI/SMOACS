import torch
from typing import Tuple


def create_equal_length_copy_mask(
    scaled_batch_abc: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Create a copy matrix for lattice lengths that are (almost) equal.

    For each structure n and each axis j in {0, 1, 2}, this function:

    1. Finds all axes k such that ``|scaled_batch_abc[n, j] - scaled_batch_abc[n, k]| < eps``.
    2. Picks the smallest such index k as the *representative* axis for j.
    3. Sets ``copy_mask[n, j, k] = 1`` if k is the representative, and 0 otherwise.

    If the original lattice lengths are ``abc = (a, b, c)``, then

        new_abc = copy_mask @ abc

    copies the length from the representative axis. For example, if ``a ≈ b != c``
    (a = b within tolerance), the mask is

        [[1, 0, 0],
         [1, 0, 0],
         [0, 0, 1]]

    and ``(a, b, c) -> (a, a, c)``.

    Args:
        scaled_batch_abc: Tensor of shape ``(N, 3)`` with (normalized) lattice lengths.
        eps: Tolerance used to decide whether two lengths are equal.

    Returns:
        Tensor of shape ``(N, 3, 3)``. Each [n] is a 3x3 matrix with one 1 per row.
    """
    device = scaled_batch_abc.device
    dtype = scaled_batch_abc.dtype
    N = scaled_batch_abc.size(0)

    if N == 0:
        return torch.zeros((0, 3, 3), device=device, dtype=dtype)

    # diff[n, j, k] = abc[n, j] - abc[n, k]
    diff = scaled_batch_abc.unsqueeze(-1) - scaled_batch_abc.unsqueeze(-2)
    eq = (diff.abs() < eps)  # (N, 3, 3) bool

    # For each (n, j), choose the smallest k with eq[n, j, k] == True
    idx = torch.arange(3, device=device).view(1, 1, 3)  # (1, 1, 3) -> (N, 3, 3)
    large = torch.full_like(idx, 10_000)
    masked_idx = torch.where(eq, idx, large)  # (N, 3, 3)
    src_idx = masked_idx.min(dim=-1).values  # (N, 3)

    copy_mask = torch.zeros((N, 3, 3), device=device, dtype=dtype)
    batch_idx = torch.arange(N, device=device).view(N, 1).expand(N, 3)
    row_idx = torch.arange(3, device=device).view(1, 3).expand(N, 3)
    copy_mask[batch_idx, row_idx, src_idx] = 1.0

    return copy_mask


def enforce_equal_lattice_lengths(
    batch_abc: torch.Tensor,
    copy_mask: torch.Tensor,
) -> torch.Tensor:
    """Enforce equal lattice lengths by copying from representative axes.

    This function assumes that ``copy_mask`` was created by
    :func:`create_equal_length_copy_mask`. The new lattice lengths are computed as

        new_abc = copy_mask @ batch_abc

    which overwrites lattice components that belong to the same equality group
    with the value of their representative axis.

    Args:
        batch_abc: Tensor of shape ``(N, 3)`` with lattice lengths to be modified.
        copy_mask: Tensor of shape ``(N, 3, 3)``. Each [n] is a 3x3 copy matrix.

    Returns:
        Tensor of shape ``(N, 3)`` with updated lattice lengths.
    """
    if batch_abc.numel() == 0:
        return batch_abc
    return torch.bmm(copy_mask, batch_abc.unsqueeze(-1)).squeeze(-1)


def create_special_angle_fix_mask(
    batch_angle: torch.Tensor,
    eps: float = 1e-3,
    special_angles: Tuple[float, float, float] = (60.0, 90.0, 120.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create mask and fixed-angle tensor for snapping angles to 60/90/120 degrees.

    For each angle entry, we check whether it is within ``eps`` of any of the
    ``special_angles`` (60°, 90°, 120° by default). If so, that entry will be
    *fixed* to the closest special angle, and its gradient will be blocked.

    Args:
        batch_angle: Tensor of shape ``(N, 3)`` with lattice angles in degrees.
        eps: Tolerance for deciding whether an angle should be snapped.
        special_angles: Tuple of special angles (in degrees) to snap to.

    Returns:
        mask: Tensor of shape ``(N, 3)`` (float).
            Entries are 1.0 where the original angle should be kept (gradient flows),
            and 0.0 where the angle should be fixed to a special value.
        fixed_angle_tensor: Tensor of shape ``(N, 3)`` (float).
            At positions where ``mask == 0``, this tensor contains the special
            angle (60/90/120); elsewhere it is zero and will be ignored.
    """
    dtype = batch_angle.dtype

    if batch_angle.numel() == 0:
        return batch_angle.new_ones(batch_angle.shape), batch_angle.new_zeros(batch_angle.shape)

    is_60 = (batch_angle - special_angles[0]).abs() < eps
    is_90 = (batch_angle - special_angles[1]).abs() < eps
    is_120 = (batch_angle - special_angles[2]).abs() < eps

    is_fixed = is_60 | is_90 | is_120  # (N, 3) bool

    fixed_angle_tensor = (
        is_60.to(dtype) * special_angles[0]
        + is_90.to(dtype) * special_angles[1]
        + is_120.to(dtype) * special_angles[2]
    )

    mask = (~is_fixed).to(dtype)

    return mask, fixed_angle_tensor


def fix_special_lattice_angles(
    batch_angle: torch.Tensor,
    mask: torch.Tensor,
    fixed_angle_tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply a snapping mask to lattice angles.

    The fixed angles are applied as

        out = batch_angle * mask + fixed_angle_tensor

    so that positions with ``mask == 1`` keep their original value and gradient,
    while positions with ``mask == 0`` are replaced by the corresponding
    special angle and do not receive gradients.

    Args:
        batch_angle: Tensor of shape ``(N, 3)`` with original lattice angles.
        mask: Tensor of shape ``(N, 3)`` as returned by
            :func:`create_special_angle_fix_mask`.
        fixed_angle_tensor: Tensor of shape ``(N, 3)`` with special angles.

    Returns:
        Tensor of shape ``(N, 3)`` with some entries snapped to special angles.
    """
    if batch_angle.numel() == 0:
        return batch_angle
    return batch_angle * mask + fixed_angle_tensor


# In[71]:


import torch

def test_abc_helpers():
    # scaled values used only for equality detection
    scaled_batch_abc = torch.tensor([
        [3.0, 3.0, 5.0],  # a = b != c
        [4.0, 6.0, 4.0],  # a = c != b
        [2.0, 7.0, 9.0],  # all different
    ], dtype=torch.float32)

    # actual values that will be modified
    batch_abc = torch.tensor([
        [3.1, 2.9, 5.0],
        [4.0, 6.2, 3.8],
        [2.0, 7.0, 9.0],
    ], dtype=torch.float32)

    expected_copy_mask = torch.tensor([
        # a = b != c
        [[1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]],
        # a = c != b
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0]],
        # all different -> identity
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
    ], dtype=torch.float32)

    expected_new_abc = torch.tensor([
        [3.1, 3.1, 5.0],  # b is overwritten by a
        [4.0, 6.2, 4.0],  # c is overwritten by a
        [2.0, 7.0, 9.0],  # unchanged
    ], dtype=torch.float32)

    copy_mask = create_equal_length_copy_mask(scaled_batch_abc, eps=1e-5)
    new_abc = enforce_equal_lattice_lengths(batch_abc, copy_mask)

    assert torch.allclose(copy_mask, expected_copy_mask), "copy_mask mismatch"
    assert torch.allclose(new_abc, expected_new_abc), "new_abc mismatch"
    print("ABC helper tests passed.")


def test_angle_helpers():
    batch_angle = torch.tensor([
        [90.0,     89.9995, 100.0],
        [60.0004, 120.0009,  45.0],
    ], dtype=torch.float32)

    # With eps=1e-3:
    #   90.0      -> 90 (fixed)
    #   89.9995   -> 90 (fixed)
    #   60.0004   -> 60 (fixed)
    #   120.0009  -> 120 (fixed)
    #   100, 45   -> unchanged
    expected_mask = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    expected_fixed = torch.tensor([
        [90.0, 90.0,  0.0],
        [60.0,120.0,  0.0],
    ], dtype=torch.float32)

    expected_fixed_angle = torch.tensor([
        [90.0, 90.0, 100.0],
        [60.0,120.0,  45.0],
    ], dtype=torch.float32)

    mask, fixed_tensor = create_special_angle_fix_mask(batch_angle, eps=1e-3)

    assert torch.allclose(mask, expected_mask), "angle mask mismatch"
    assert torch.allclose(fixed_tensor, expected_fixed), "fixed angle tensor mismatch"

    fixed_angle = fix_special_lattice_angles(batch_angle, mask, fixed_tensor)
    assert torch.allclose(fixed_angle, expected_fixed_angle), "fixed_angle result mismatch"

    print("Angle helper tests passed.")


if __name__ == "__main__":
    test_abc_helpers()
    test_angle_helpers()