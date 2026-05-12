# src/spg_wp/R3m.py
"""Wyckoff positions for space group R3m (No. 160, hexagonal axes).

This module provides a class-based container for the Wyckoff data of R3m.
Each Wyckoff position is defined by an explicit set of affine operators
(mul, add) such that

    r' = M @ r + t

with ``M`` of shape (3, 3) and ``t`` of shape (3,).

Typical usage:
    from R3m import R3mWyckoff

    r3m = R3mWyckoff.build()
    table = r3m.wyckoff_table
    dup   = r3m.dup_pos
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class R3mWyckoff:
    """Container for Wyckoff data of space group R3m (No. 160).

    This uses the hexagonal setting with rhombohedral centering vectors.

    Attributes:
        spg_number: International number of the space group (160).
        wyckoff_table: Mapping from Wyckoff ID to (mul, add) tensors.
            ``mul`` has shape (k, 3, 3), ``add`` has shape (k, 3).
        wyckoff_site: List of Wyckoff labels ordered by ID
            (["3a", "9b", "18c"]).
        num_equivalent_sites: List of multiplicities for each Wyckoff ID.
        dup_pos: Tensor of centering translations of shape (D, 3).
            For R3m (R-lattice), this contains the three rhombohedral
            centering vectors in hexagonal coordinates:

                (0, 0, 0)
                (2/3, 1/3, 1/3)
                (1/3, 2/3, 2/3)
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
    def _ops_18c() -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 6 general-position operators for R3m (Wyckoff 18c).

        Coordinate forms in fractional (x, y, z) for the non-centered
        general position (hexagonal axes)::

            (x,       y,       z)
            (-y,      -x,      z)
            (-x + y,  -x,      z)
            (-y,      x - y,   z)
            (-x + y,  y,       z)
            (x,       x - y,   z)

        The full 18c position is obtained by applying the three
        rhombohedral centering translations to these six positions.

        Each operator is represented as a 3x3 matrix ``M`` and a
        translation vector ``t`` such that ``p' = M @ p + t``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Ms: Tensor of shape (6, 3, 3), linear-part matrices.
                - Ts: Tensor of shape (6, 3), translation vectors
                  (all zeros for 18c).
        """
        Ms: List[torch.Tensor] = []
        Ts: List[torch.Tensor] = []

        # (x, y, z)
        Ms.append(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

        # (-y, -x, z)
        Ms.append(
            torch.tensor(
                [
                    [0.0, -1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

        # (-x + y, -x, z)
        Ms.append(
            torch.tensor(
                [
                    [-1.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

        # (-y, x - y, z)
        Ms.append(
            torch.tensor(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

        # (-x + y, y, z)
        Ms.append(
            torch.tensor(
                [
                    [-1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

        # (x, x - y, z)
        Ms.append(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        )

        for _ in range(6):
            Ts.append(torch.zeros(3, dtype=torch.float32))

        return torch.stack(Ms, dim=0), torch.stack(Ts, dim=0)

    # ------------------------------------------------------------------ #
    # Constructor                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls) -> "R3mWyckoff":
        """Build a :class:`R3mWyckoff` instance with all data populated.

        This method defines the affine operators for each Wyckoff position
        of R3m (3a, 9b, 18c) in a straightforward, explicit way. The
        global rhombohedral centering is encoded in ``dup_pos``; all
        Wyckoff positions share the same centering translations.

        Returns:
            R3mWyckoff: Fully constructed Wyckoff data for R3m.
        """
        # 3a : (0, 0, z)   multiplicity 3 (1 op × 3 centerings)
        # Only the z-component is used; x, y are ignored.
        M_a = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],  # x' = 0
                    [0.0, 0.0, 0.0],  # y' = 0
                    [0.0, 0.0, 1.0],  # z' = z
                ],
            ],
            dtype=torch.float32,
        )
        t_a = torch.zeros((1, 3), dtype=torch.float32)

        # 9b : (x, -x, z), (x, 2x, z), (-2x, -x, z)   multiplicity 9
        # Only the first coordinate (x-parameter) and z are used.
        M_b = torch.tensor(
            [
                # (x, -x, z)
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                # (x, 2x, z)
                [
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                # (-2x, -x, z)
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )
        t_b = torch.zeros((3, 3), dtype=torch.float32)

        # 18c : general position (x, y, z) with 6 operations
        M_c, t_c = cls._ops_18c()

        wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
            0: (M_a, t_a),  # 3a
            1: (M_b, t_b),  # 9b
            2: (M_c, t_c),  # 18c
        }

        wyckoff_site: List[str] = ["3a", "9b", "18c"]

        # Rhombohedral centering vectors in hexagonal axes.
        dup_pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
            ],
            dtype=torch.float32,
        )

        num_equiv = [
            wyckoff_table[i][0].shape[0] * dup_pos.shape[0]
            for i in range(len(wyckoff_table))
        ]

        return cls(
            spg_number=160,
            wyckoff_table=wyckoff_table,
            wyckoff_site=wyckoff_site,
            num_equivalent_sites=num_equiv,
            dup_pos=dup_pos,
        )

