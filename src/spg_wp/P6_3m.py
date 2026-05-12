# P6_3m.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class P63mWyckoff:
    """Container for Wyckoff data of space group P6_3/m (No. 176).

    Attributes:
        spg_number: International number of the space group (176).
        wyckoff_table: Mapping from Wyckoff ID to (mul, add) tensors.
            ``mul`` has shape (k, 3, 3), ``add`` has shape (k, 3).
        wyckoff_site: List of Wyckoff labels ordered by ID
            (e.g., ["2a", "2b", "2c", ...]).
        num_equivalent_sites: List of multiplicities for each Wyckoff ID.
        dup_pos: Tensor of centering translations of shape (D, 3).
            For P6_3/m (P-lattice), this is just ``[[0, 0, 0]]``.
    """

    spg_number: int
    wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    wyckoff_site: List[str]
    num_equivalent_sites: List[int]
    dup_pos: torch.Tensor

    # ------------------------------------------------------------------ #
    # General position 12i : (x, y, z) with 12 symmetry operations
    #   x,y,z
    #   -y, x-y, z
    #   -x+y, -x, z
    #   -x, -y, z+1/2
    #   y, -x+y, z+1/2
    #   x-y, x, z+1/2
    #   -x, -y, -z
    #   y, -x+y, -z
    #   x-y, x, -z
    #   x, y, -z+1/2
    #   -y, x-y, -z+1/2
    #   -x+y, -x, -z+1/2
    # (ITA / yseto table for P6_3/m):contentReference[oaicite:1]{index=1}
    # ------------------------------------------------------------------ #
    @classmethod
    def _ops_12i(cls) -> Tuple[torch.Tensor, torch.Tensor]:
        M = torch.tensor(
            [
                # 1: (x, y, z)
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # 2: (-y, x - y, z)
                [[0.0, -1.0, 0.0],
                 [1.0, -1.0, 0.0],
                 [0.0,  0.0, 1.0]],
                # 3: (-x + y, -x, z)
                [[-1.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                # 4: (-x, -y, z + 1/2)
                [[-1.0, 0.0, 0.0],
                 [ 0.0,-1.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                # 5: (y, -x + y, z + 1/2)
                [[0.0, 1.0, 0.0],
                 [-1.0, 1.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                # 6: (x - y, x, z + 1/2)
                [[1.0,-1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # 7: (-x, -y, -z)
                [[-1.0, 0.0,  0.0],
                 [ 0.0,-1.0,  0.0],
                 [ 0.0, 0.0, -1.0]],
                # 8: (y, -x + y, -z)
                [[0.0, 1.0,  0.0],
                 [-1.0,1.0,  0.0],
                 [0.0, 0.0, -1.0]],
                # 9: (x - y, x, -z)
                [[1.0,-1.0,  0.0],
                 [1.0, 0.0,  0.0],
                 [0.0, 0.0, -1.0]],
                #10: (x, y, -z + 1/2)
                [[1.0, 0.0,  0.0],
                 [0.0, 1.0,  0.0],
                 [0.0, 0.0, -1.0]],
                #11: (-y, x - y, -z + 1/2)
                [[0.0,-1.0,  0.0],
                 [1.0,-1.0,  0.0],
                 [0.0, 0.0, -1.0]],
                #12: (-x + y, -x, -z + 1/2)
                [[-1.0, 1.0,  0.0],
                 [-1.0, 0.0,  0.0],
                 [ 0.0, 0.0, -1.0]],
            ],
            dtype=torch.float32,
        )

        t = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # 1
                [0.0, 0.0, 0.0],  # 2
                [0.0, 0.0, 0.0],  # 3
                [0.0, 0.0, 0.5],  # 4
                [0.0, 0.0, 0.5],  # 5
                [0.0, 0.0, 0.5],  # 6
                [0.0, 0.0, 0.0],  # 7
                [0.0, 0.0, 0.0],  # 8
                [0.0, 0.0, 0.0],  # 9
                [0.0, 0.0, 0.5],  #10
                [0.0, 0.0, 0.5],  #11
                [0.0, 0.0, 0.5],  #12
            ],
            dtype=torch.float32,
        )

        return M, t

    # ------------------------------------------------------------------ #
    # Builder
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls) -> "P63mWyckoff":
        """Build a :class:`P63mWyckoff` instance with all data populated."""

        # 2a : (0, 0, 1/4); (0, 0, 3/4)
        M_a = torch.zeros((2, 3, 3), dtype=torch.float32)
        t_a = torch.tensor(
            [
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 0.75],
            ],
            dtype=torch.float32,
        )

        # 2b : (0, 0, 0); (0, 0, 1/2)
        M_b = torch.zeros((2, 3, 3), dtype=torch.float32)
        t_b = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )

        # 2c : (1/3, 2/3, 1/4); (2/3, 1/3, 3/4)
        M_c = torch.zeros((2, 3, 3), dtype=torch.float32)
        t_c = torch.tensor(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.25],
                [2.0 / 3.0, 1.0 / 3.0, 0.75],
            ],
            dtype=torch.float32,
        )

        # 2d : (2/3, 1/3, 1/4); (1/3, 2/3, 3/4)
        M_d = torch.zeros((2, 3, 3), dtype=torch.float32)
        t_d = torch.tensor(
            [
                [2.0 / 3.0, 1.0 / 3.0, 0.25],
                [1.0 / 3.0, 2.0 / 3.0, 0.75],
            ],
            dtype=torch.float32,
        )

        # 4e : (0,0,z); (0,0,z+1/2); (0,0,-z); (0,0,-z+1/2)
        M_e = torch.tensor(
            [
                # (0, 0,  z)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # (0, 0,  z+1/2)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # (0, 0, -z)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0,-1.0]],
                # (0, 0, -z+1/2)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0,-1.0]],
            ],
            dtype=torch.float32,
        )
        t_e = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # z
                [0.0, 0.0, 0.5],  # z+1/2
                [0.0, 0.0, 0.0],  # -z
                [0.0, 0.0, 0.5],  # -z+1/2
            ],
            dtype=torch.float32,
        )

        # 4f : (1/3,2/3,z); (2/3,1/3,z+1/2); (2/3,1/3,-z); (1/3,2/3,-z+1/2)
        M_f = M_e.clone()
        t_f = torch.tensor(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.0],   # +z
                [2.0 / 3.0, 1.0 / 3.0, 0.5],  # +z+1/2
                [2.0 / 3.0, 1.0 / 3.0, 0.0],  # -z
                [1.0 / 3.0, 2.0 / 3.0, 0.5],  # -z+1/2
            ],
            dtype=torch.float32,
        )

        # 6g : (1/2,0,0); (0,1/2,0); (1/2,1/2,0);
        #       (1/2,0,1/2); (0,1/2,1/2); (1/2,1/2,1/2)
        M_g = torch.zeros((6, 3, 3), dtype=torch.float32)
        t_g = torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )

        # 6h : (x,y,1/4); (-y,x-y,1/4); (-x+y,-x,1/4);
        #      (-x,-y,3/4); (y,-x+y,3/4); (x-y,x,3/4)
        M_h = torch.tensor(
            [
                # (x, y, 1/4)
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0]],
                # (-y, x-y, 1/4)
                [[0.0,-1.0, 0.0],
                 [1.0,-1.0, 0.0],
                 [0.0, 0.0, 0.0]],
                # (-x+y, -x, 1/4)
                [[-1.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0]],
                # (-x, -y, 3/4)
                [[-1.0, 0.0, 0.0],
                 [ 0.0,-1.0, 0.0],
                 [ 0.0, 0.0, 0.0]],
                # (y, -x+y, 3/4)
                [[0.0, 1.0, 0.0],
                 [-1.0,1.0, 0.0],
                 [0.0, 0.0, 0.0]],
                # (x-y, x, 3/4)
                [[1.0,-1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        t_h = torch.tensor(
            [
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 0.75],
            ],
            dtype=torch.float32,
        )

        # 12i : general position
        M_i, t_i = cls._ops_12i()

        wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
            0: (M_a, t_a),   # 2a
            1: (M_b, t_b),   # 2b
            2: (M_c, t_c),   # 2c
            3: (M_d, t_d),   # 2d
            4: (M_e, t_e),   # 4e
            5: (M_f, t_f),   # 4f
            6: (M_g, t_g),   # 6g
            7: (M_h, t_h),   # 6h
            8: (M_i, t_i),   # 12i
        }

        wyckoff_site: List[str] = [
            "2a", "2b", "2c", "2d", "4e", "4f", "6g", "6h", "12i"
        ]

        dup_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        num_equiv = [
            wyckoff_table[i][0].shape[0] * dup_pos.shape[0]
            for i in range(len(wyckoff_table))
        ]

        return cls(
            spg_number=176,
            wyckoff_table=wyckoff_table,
            wyckoff_site=wyckoff_site,
            num_equivalent_sites=num_equiv,
            dup_pos=dup_pos,
        )

