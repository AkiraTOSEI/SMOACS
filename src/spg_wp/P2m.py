# src/spg_wp/P2m.py
"""Wyckoff positions for space group P2/m (No. 10, unique axis b).

This module provides a class-based container for the Wyckoff data of P2/m.
Each Wyckoff position is defined by an explicit set of affine operators
(mul, add) such that

    r' = M @ r + t

with ``M`` of shape (3, 3) and ``t`` of shape (3,).

Typical usage:
    from P2m import P2mWyckoff

    p2m = P2mWyckoff.build()
    table = p2m.wyckoff_table
    dup   = p2m.dup_pos
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class P2mWyckoff:
    """Container for Wyckoff data of space group P2/m (No. 10).

    Attributes:
        spg_number: International number of the space group (10).
        wyckoff_table: Mapping from Wyckoff ID to (mul, add) tensors.
            ``mul`` has shape (k, 3, 3), ``add`` has shape (k, 3).
        wyckoff_site: List of Wyckoff labels ordered by ID
            (e.g., ["1a", "1b", "1c", ...]).
        num_equivalent_sites: List of multiplicities for each Wyckoff ID.
        dup_pos: Tensor of centering translations of shape (D, 3).
            For P2/m (P-lattice), this is just ``[[0, 0, 0]]``.
    """

    spg_number: int
    wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    wyckoff_site: List[str]
    num_equivalent_sites: List[int]
    dup_pos: torch.Tensor

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ops_4o() -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 4 general-position operators for P2/m (Wyckoff 4o).

        Unique axis b, coordinates in fractional (x, y, z)::

            (x,  y,  z)
            (x, -y,  z)
            (-x, y, -z)
            (-x,-y, -z)

        Each operator is represented as a 3x3 matrix ``M`` and a translation
        vector ``t`` such that ``p' = M @ p + t``.
        """
        Ms: List[torch.Tensor] = []
        Ts: List[torch.Tensor] = []

        # (x, y, z)
        Ms.append(
            torch.tensor(
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        )

        # (x, -y, z)
        Ms.append(
            torch.tensor(
                [[1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        )

        # (-x, y, -z)
        Ms.append(
            torch.tensor(
                [[-1.0, 0.0, 0.0],
                 [ 0.0, 1.0, 0.0],
                 [ 0.0, 0.0,-1.0]],
                dtype=torch.float32,
            )
        )

        # (-x, -y, -z)
        Ms.append(
            torch.tensor(
                [[-1.0, 0.0, 0.0],
                 [ 0.0,-1.0, 0.0],
                 [ 0.0, 0.0,-1.0]],
                dtype=torch.float32,
            )
        )

        for _ in range(4):
            Ts.append(torch.zeros(3, dtype=torch.float32))

        return torch.stack(Ms, dim=0), torch.stack(Ts, dim=0)

    # ------------------------------------------------------------------ #
    # Constructor                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls) -> "P2mWyckoff":
        """Build a :class:`P2mWyckoff` instance with all data populated.

        Wyckoff positions (unique axis b) covered here:

            1a–1h, 2i–2n, 4o

        The operators are written explicitly; there is no automatic
        symmetry-detection logic.
        """
        # ------------------------------------------------------------------
        # 1a–1h : fixed points (site symmetry 2/m), multiplicity 1
        # ------------------------------------------------------------------
        # All coordinates here are constants, so the linear part is zero.
        M_1x = torch.zeros((1, 3, 3), dtype=torch.float32)

        # 1a : (0, 0, 0)
        M_a = M_1x.clone()
        t_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        # 1b : (0, 1/2, 0)
        M_b = M_1x.clone()
        t_b = torch.tensor([[0.0, 0.5, 0.0]], dtype=torch.float32)

        # 1c : (0, 0, 1/2)
        M_c = M_1x.clone()
        t_c = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)

        # 1d : (1/2, 0, 0)
        M_d = M_1x.clone()
        t_d = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)

        # 1e : (1/2, 1/2, 0)
        M_e = M_1x.clone()
        t_e = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)

        # 1f : (0, 1/2, 1/2)
        M_f = M_1x.clone()
        t_f = torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float32)

        # 1g : (1/2, 0, 1/2)
        M_g = M_1x.clone()
        t_g = torch.tensor([[0.5, 0.0, 0.5]], dtype=torch.float32)

        # 1h : (1/2, 1/2, 1/2)
        M_h = M_1x.clone()
        t_h = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)

        # ------------------------------------------------------------------
        # 2i–2l : points on 2-fold axis (x = 0 or 1/2, z = 0 or 1/2)
        # site symmetry 2, multiplicity 2
        # General forms (unique axis b):
        #   2i : (0,  y, 0), (0, -y, 0)
        #   2j : (1/2, y, 0), (1/2, -y, 0)
        #   2k : (0,  y, 1/2), (0, -y, 1/2)
        #   2l : (1/2, y, 1/2), (1/2, -y, 1/2)
        # Only y is a free parameter.
        # ------------------------------------------------------------------
        M_i = torch.tensor(
            [
                # (0,  y, 0)
                [[0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0]],
                # (0, -y, 0)
                [[0.0, 0.0, 0.0],
                 [0.0,-1.0, 0.0],
                 [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        t_i = torch.zeros((2, 3), dtype=torch.float32)

        # 2j : shift x by 1/2
        M_j = M_i.clone()
        t_j = torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        # 2k : shift z by 1/2
        M_k = M_i.clone()
        t_k = torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )

        # 2l : shift x and z by 1/2
        M_l = M_i.clone()
        t_l = torch.tensor(
            [
                [0.5, 0.0, 0.5],
                [0.5, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # 2m, 2n : mirror planes y = 0 and y = 1/2 (site symmetry m)
        #   2m : (x, 0, z), (-x, 0, -z)
        #   2n : (x, 1/2, z), (-x, 1/2, -z)
        # Free parameters: x, z
        # ------------------------------------------------------------------
        M_m = torch.tensor(
            [
                # (x, 0, z)
                [[ 1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                # (-x, 0, -z)
                [[-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0],
                 [ 0.0, 0.0,-1.0]],
            ],
            dtype=torch.float32,
        )
        t_m = torch.zeros((2, 3), dtype=torch.float32)

        M_n = M_m.clone()
        t_n = torch.tensor(
            [
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )

        # ------------------------------------------------------------------
        # 4o : general position (site symmetry 1), multiplicity 4
        # ------------------------------------------------------------------
        M_o, t_o = cls._ops_4o()

        # ------------------------------------------------------------------
        # Assemble tables
        # ------------------------------------------------------------------
        wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
            0: (M_a, t_a),  # 1a
            1: (M_b, t_b),  # 1b
            2: (M_c, t_c),  # 1c
            3: (M_d, t_d),  # 1d
            4: (M_e, t_e),  # 1e
            5: (M_f, t_f),  # 1f
            6: (M_g, t_g),  # 1g
            7: (M_h, t_h),  # 1h
            8: (M_i, t_i),  # 2i
            9: (M_j, t_j),  # 2j
            10: (M_k, t_k),  # 2k
            11: (M_l, t_l),  # 2l
            12: (M_m, t_m),  # 2m
            13: (M_n, t_n),  # 2n
            14: (M_o, t_o),  # 4o
        }

        wyckoff_site: List[str] = [
            "1a",
            "1b",
            "1c",
            "1d",
            "1e",
            "1f",
            "1g",
            "1h",
            "2i",
            "2j",
            "2k",
            "2l",
            "2m",
            "2n",
            "4o",
        ]

        # P-lattice: only identity centering.
        dup_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        num_equiv = [
            wyckoff_table[i][0].shape[0] * dup_pos.shape[0]
            for i in range(len(wyckoff_table))
        ]

        return cls(
            spg_number=10,
            wyckoff_table=wyckoff_table,
            wyckoff_site=wyckoff_site,
            num_equivalent_sites=num_equiv,
            dup_pos=dup_pos,
        )

