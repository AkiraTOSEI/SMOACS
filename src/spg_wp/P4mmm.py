from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


Tensor = torch.Tensor
WyckoffTable = Dict[int, Tuple[Tensor, Tensor]]  # {id: (mul, add)}


@dataclass
class P4mmmWyckoff:
    """Wyckoff data container for space group P4/mmm (No. 123).

    Attributes
    ----------
    spg_number:
        International space-group number (123).

    wyckoff_table:
        Mapping from Wyckoff *ID* (0-based) to a pair of tensors (M, t).

        * M: shape (k, 3, 3)
        * t: shape (k, 3)

        Each operation corresponds to

            x' = M @ p + t

        where ``p = (x, y, z)`` is a 3‑component parameter vector
        (only a subset of its components is actually used for a given
        Wyckoff position, exactly like in ``P321Wyckoff``).

    wyckoff_site:
        List of Wyckoff symbols ordered by ID, e.g. ``["1a", "1b", ...]``.

    num_equivalent_sites:
        Multiplicity for each Wyckoff ID, i.e. number of equivalent sites
        per asymmetric unit (for a P‑lattice this is just
        ``mul_tensor.shape[0]``).

    dup_pos:
        Tensor of centering translations of shape (D, 3).
        For P4/mmm (primitive tetragonal), this is just ``[[0, 0, 0]]``.
    """

    spg_number: int
    wyckoff_table: WyckoffTable
    wyckoff_site: List[str]
    num_equivalent_sites: List[int]
    dup_pos: Tensor

    # ------------------------------------------------------------------
    # Core building utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _symmetry_rotations() -> Tensor:
        """Return the 16 rotation matrices of P4/mmm general position.

        Operations are taken from the cctbx P4/mmm listing and ordered
        exactly as there (x,y,z), (-y,x,z), (-x,-y,z), ... .
        """
        R = torch.tensor(
            [
                # 1: x,  y,  z
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # 2: -y, x,  z
                [[0.0, -1.0, 0.0],
                 [1.0,  0.0, 0.0],
                 [0.0,  0.0, 1.0]],
                # 3: -x,-y,  z
                [[-1.0, 0.0, 0.0],
                 [ 0.0,-1.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                # 4:  y,-x,  z
                [[0.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # 5:  x,-y,-z
                [[1.0, 0.0, 0.0],
                 [0.0,-1.0, 0.0],
                 [0.0, 0.0,-1.0]],
                # 6:  y, x,-z
                [[0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0,-1.0]],
                # 7: -x, y,-z
                [[-1.0, 0.0, 0.0],
                 [ 0.0, 1.0, 0.0],
                 [ 0.0, 0.0,-1.0]],
                # 8: -y,-x,-z
                [[ 0.0,-1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 0.0, 0.0,-1.0]],
                # 9: -x,-y,-z
                [[-1.0, 0.0, 0.0],
                 [ 0.0,-1.0, 0.0],
                 [ 0.0, 0.0,-1.0]],
                # 10: y,-x,-z
                [[0.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [0.0, 0.0,-1.0]],
                # 11: x, y,-z
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0,-1.0]],
                # 12: -y, x,-z
                [[ 0.0,-1.0, 0.0],
                 [ 1.0, 0.0, 0.0],
                 [ 0.0, 0.0,-1.0]],
                # 13: -x, y, z
                [[-1.0, 0.0, 0.0],
                 [ 0.0, 1.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                # 14: -y,-x, z
                [[ 0.0,-1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                # 15: x,-y, z
                [[1.0, 0.0, 0.0],
                 [0.0,-1.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # 16: y, x,  z
                [[0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        return R

    @classmethod
    def _build_site(
        cls,
        R: Tensor,
        B: Tensor,
        t0: Tensor,
        multiplicity: int,
    ) -> Tuple[Tensor, Tensor]:
        """Construct (M, t) for one Wyckoff position.

        Parameters
        ----------
        R:
            (16, 3, 3) rotation matrices for the general position.
        B:
            3×3 matrix that encodes how the free parameters (x, y, z)
            enter the representative position, e.g.

                * general 16u:   B = I
                * 8t (x,1/2,z):  B = diag(1,0,1)
                * 4j (x,x,0):    B = [[1,0,0],[1,0,0],[0,0,0]]

        t0:
            Constant shift of the representative position
            (e.g. (0, 1/2, 0)).

        multiplicity:
            Expected number of distinct equivalent positions for this
            Wyckoff site.

        Returns
        -------
        M_site, t_site:
            Tensors with shapes (multiplicity, 3, 3) and (multiplicity, 3)
            respectively.
        """
        B = B.to(dtype=torch.float32)
        t0 = t0.to(dtype=torch.float32)

        # All 16 affine operators for this representative position
        # x' = R * (B p + t0) = (R B) p + (R t0).
        M_all = torch.matmul(R, B)                      # (16, 3, 3)
        t0_col = t0.view(3, 1)
        t_all = torch.matmul(R, t0_col).squeeze(-1)     # (16, 3)

        # Use a "generic" parameter vector to detect which of the 16
        # operators produce identical positions.
        params = torch.tensor([0.173, 0.286, 0.413], dtype=torch.float32)

        coords = (torch.matmul(M_all, params) + t_all) % 1.0  # (16, 3)

        # Unique positions (modulo 1) and map back to the first index
        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
        num_unique = int(unique_coords.shape[0])

        indices = torch.full((num_unique,), -1, dtype=torch.long)
        for orig_idx, uidx in enumerate(inverse):
            j = int(uidx.item())
            if indices[j] < 0:
                indices[j] = orig_idx

        assert (indices >= 0).all()
        assert (
            num_unique == multiplicity
        ), f"Multiplicity mismatch: expected {multiplicity}, got {num_unique}"

        M_site = M_all[indices]
        t_site = t_all[indices]

        return M_site, t_site

    # ------------------------------------------------------------------
    # Public constructor
    # ------------------------------------------------------------------

    @classmethod
    def build(cls) -> "P4mmmWyckoff":
        """Build Wyckoff data for P4/mmm (No. 123).

        The Wyckoff positions follow the standard sequence

            1a, 1b, 1c, 1d,
            2e, 2f, 2g, 2h,
            4i, 4j, 4k, 4l, 4m, 4n, 4o,
            8p, 8q, 8r, 8s, 8t,
            16u

        with multiplicities and representatives taken from
        crystallographic tables (cctbx / International Tables).
        """
        R = cls._symmetry_rotations()

        def zeros3x3() -> Tensor:
            return torch.zeros((3, 3), dtype=torch.float32)

        def diag3(a: float, b: float, c: float) -> Tensor:
            return torch.diag(torch.tensor([a, b, c], dtype=torch.float32))

        wyckoff_table: WyckoffTable = {}
        wyckoff_site: List[str] = []
        mult_list: List[int] = []

        # ------------------------------------------------------------------
        # 1-fold positions (1a–1d)
        # ------------------------------------------------------------------
        onefold_specs = [
            ("1a", 1, zeros3x3(), torch.tensor([0.0, 0.0, 0.0])),
            ("1b", 1, zeros3x3(), torch.tensor([0.0, 0.0, 0.5])),
            ("1c", 1, zeros3x3(), torch.tensor([0.5, 0.5, 0.0])),
            ("1d", 1, zeros3x3(), torch.tensor([0.5, 0.5, 0.5])),
        ]

        # 2-fold positions (2e–2h)
        twofold_specs = [
            # 2e : (0, 1/2, 1/2)
            ("2e", 2, zeros3x3(), torch.tensor([0.0, 0.5, 0.5])),
            # 2f : (0, 1/2, 0)
            ("2f", 2, zeros3x3(), torch.tensor([0.0, 0.5, 0.0])),
            # 2g : (0, 0, z)
            ("2g", 2, diag3(0.0, 0.0, 1.0), torch.tensor([0.0, 0.0, 0.0])),
            # 2h : (1/2, 1/2, z)
            ("2h", 2, diag3(0.0, 0.0, 1.0), torch.tensor([0.5, 0.5, 0.0])),
        ]

        # 4-fold positions (4i–4o)
        fourfold_specs = [
            # 4i : (0, 1/2, z)
            ("4i", 4, diag3(0.0, 0.0, 1.0), torch.tensor([0.0, 0.5, 0.0])),
            # 4j : (x, x, 0)
            ("4j", 4,
             torch.tensor([[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]], dtype=torch.float32),
             torch.tensor([0.0, 0.0, 0.0])),
            # 4k : (x, x, 1/2)
            ("4k", 4,
             torch.tensor([[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0]], dtype=torch.float32),
             torch.tensor([0.0, 0.0, 0.5])),
            ("4l", 4, diag3(1.0, 0.0, 0.0), torch.tensor([0.0, 0.0, 0.0])),  # (x, 0, 0)
            ("4m", 4, diag3(1.0, 0.0, 0.0), torch.tensor([0.0, 0.0, 0.5])),  # (x, 0, 1/2)
            ("4n", 4, diag3(1.0, 0.0, 0.0), torch.tensor([0.0, 0.5, 0.0])),  # (x, 1/2, 0)
            ("4o", 4, diag3(1.0, 0.0, 0.0), torch.tensor([0.0, 0.5, 0.5])),  # (x, 1/2, 1/2)
        ]

        # 8-fold positions (8p–8t)
        eightfold_specs = [
            # 8p : (x, y, 0)
            ("8p", 8, diag3(1.0, 1.0, 0.0), torch.tensor([0.0, 0.0, 0.0])),
            # 8q : (x, y, 1/2)
            ("8q", 8, diag3(1.0, 1.0, 0.0), torch.tensor([0.0, 0.0, 0.5])),
            # 8r : (x, x, z)
            ("8r", 8,
             torch.tensor([[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]], dtype=torch.float32),
             torch.tensor([0.0, 0.0, 0.0])),
            # 8s : (x, 0, z)
            ("8s", 8, diag3(1.0, 0.0, 1.0), torch.tensor([0.0, 0.0, 0.0])),
            # 8t : (x, 1/2, z)
            ("8t", 8, diag3(1.0, 0.0, 1.0), torch.tensor([0.0, 0.5, 0.0])),
        ]

        # 16-fold general position (16u : (x, y, z))
        sixteenfold_specs = [
            ("16u", 16, torch.eye(3, dtype=torch.float32),
             torch.tensor([0.0, 0.0, 0.0])),
        ]

        all_specs = onefold_specs + twofold_specs + fourfold_specs + eightfold_specs + sixteenfold_specs

        for wy_id, (label, mult, B, t0) in enumerate(all_specs):
            M_site, t_site = cls._build_site(R, B, t0, mult)
            wyckoff_table[wy_id] = (M_site, t_site)
            wyckoff_site.append(label)
            mult_list.append(int(M_site.shape[0]))

        # Primitive lattice: only identity centering.
        dup_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        num_equiv = [
            m * int(dup_pos.shape[0]) for m in mult_list
        ]

        return cls(
            spg_number=123,
            wyckoff_table=wyckoff_table,
            wyckoff_site=wyckoff_site,
            num_equivalent_sites=num_equiv,
            dup_pos=dup_pos,
        )

