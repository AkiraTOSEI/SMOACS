# Amm2.py
"""Wyckoff positions for space group Amm2 (No. 38, orthorhombic, A-centered).

This module provides a class-based container for the Wyckoff data of Amm2.
Each Wyckoff position is defined by an explicit set of affine operators
(mul, add) such that

    r' = M @ r + t

with ``M`` of shape (3, 3) and ``t`` of shape (3,).

Typical usage:
    from Amm2 import Amm2Wyckoff

    amm2 = Amm2Wyckoff.build()
    table = amm2.wyckoff_table
    dup   = amm2.dup_pos
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class Amm2Wyckoff:
    """Container for Wyckoff data of space group Amm2 (No. 38).

    Attributes:
        spg_number: International number of the space group (38).
        wyckoff_table: Mapping from Wyckoff ID to (mul, add) tensors.
            ``mul`` has shape (k, 3, 3), ``add`` has shape (k, 3).
        wyckoff_site: List of Wyckoff labels ordered by ID
            (e.g., ["2a", "2b", "4c", ...]).
        num_equivalent_sites: List of multiplicities for each Wyckoff ID.
        dup_pos: Tensor of centering translations of shape (D, 3).
            For Amm2 (A-lattice), this is ``[[0, 0, 0], [0, 0.5, 0.5]]``.
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
    def _ops_8f() -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 4 point-group operators for Amm2 general position (Wyckoff 8f).

        Point group mm2 has four operations before A-centering::

            (x,   y,  z)   identity
            (-x, -y,  z)   twofold rotation around c
            (-x,  y,  z)   mirror perpendicular to a
            (x,  -y,  z)   mirror perpendicular to b

        Each operator is represented as a 3x3 matrix ``M`` and a translation
        vector ``t`` such that ``p' = M @ p + t``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Ms: Tensor of shape (4, 3, 3), linear-part matrices.
                - Ts: Tensor of shape (4, 3), translation vectors
                  (all zeros for 8f).
        """
        Ms: List[torch.Tensor] = []
        Ts: List[torch.Tensor] = []

        # (x, y, z) - identity
        Ms.append(
            torch.tensor(
                [[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        )

        # (-x, -y, z) - twofold rotation around c
        Ms.append(
            torch.tensor(
                [[-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0,  0.0, 1.0]],
                dtype=torch.float32,
            )
        )

        # (-x, y, z) - mirror perpendicular to a
        Ms.append(
            torch.tensor(
                [[-1.0, 0.0, 0.0],
                 [0.0,  1.0, 0.0],
                 [0.0,  0.0, 1.0]],
                dtype=torch.float32,
            )
        )

        # (x, -y, z) - mirror perpendicular to b
        Ms.append(
            torch.tensor(
                [[1.0,  0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0,  0.0, 1.0]],
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
    def build(cls) -> "Amm2Wyckoff":
        """Build a :class:`Amm2Wyckoff` instance with all data populated.

        This method defines the affine operators for each Wyckoff position
        of Amm2 (2a–8f) in a straightforward, explicit way. No automatic
        deduplication logic is used.

        Returns:
            Amm2Wyckoff: Fully constructed Wyckoff data for Amm2.
        """
        # 2a : (0, 0, z)   multiplicity 2 (with A-centering)
        M_a = torch.tensor(
            [
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        t_a = torch.zeros((1, 3), dtype=torch.float32)

        # 2b : (1/2, 0, z)   multiplicity 2 (with A-centering)
        M_b = M_a.clone()
        t_b = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)

        # 4c : (x, 0, z), (-x, 0, z)   multiplicity 4 (with A-centering)
        # x and z vary, y = 0
        M_c = torch.tensor(
            [
                # (x, 0, z)
                [[1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # (-x, 0, z)
                [[-1.0, 0.0, 0.0],
                 [0.0,  0.0, 0.0],
                 [0.0,  0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        t_c = torch.zeros((2, 3), dtype=torch.float32)

        # 4d : (0, y, z), (0, -y, z)   multiplicity 4 (with A-centering)
        # y and z vary, x = 0
        M_d = torch.tensor(
            [
                # (0, y, z)
                [[0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # (0, -y, z)
                [[0.0,  0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0,  0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        t_d = torch.zeros((2, 3), dtype=torch.float32)

        # 4e : (1/2, y, z), (1/2, -y, z)   multiplicity 4 (with A-centering)
        # y and z vary, x = 1/2
        M_e = M_d.clone()
        t_e = torch.tensor(
            [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        # 8f : general position (x, y, z) with 4 point-group operations
        M_f, t_f = cls._ops_8f()

        wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
            0: (M_a, t_a),  # 2a
            1: (M_b, t_b),  # 2b
            2: (M_c, t_c),  # 4c
            3: (M_d, t_d),  # 4d
            4: (M_e, t_e),  # 4e
            5: (M_f, t_f),  # 8f
        }

        wyckoff_site: List[str] = ["2a", "2b", "4c", "4d", "4e", "8f"]

        # A-lattice: identity + (0, 1/2, 1/2) centering translation.
        dup_pos = torch.tensor(
            [[0.0, 0.0, 0.0],
             [0.0, 0.5, 0.5]],
            dtype=torch.float32
        )

        num_equiv = [
            wyckoff_table[i][0].shape[0] * dup_pos.shape[0]
            for i in range(len(wyckoff_table))
        ]

        return cls(
            spg_number=38,
            wyckoff_table=wyckoff_table,
            wyckoff_site=wyckoff_site,
            num_equivalent_sites=num_equiv,
            dup_pos=dup_pos,
        )
