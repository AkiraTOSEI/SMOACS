# src/spg_wp/P321.py
"""Wyckoff positions for space group P321 (No. 150, hexagonal axes).

This module provides a class-based container for the Wyckoff data of P321.
Each Wyckoff position is defined by an explicit set of affine operators
(mul, add) such that

    r' = M @ r + t

with ``M`` of shape (3, 3) and ``t`` of shape (3,).

Typical usage:
    from src.spg_wp.P321 import P321Wyckoff

    p321 = P321Wyckoff.build()
    table = p321.wyckoff_table
    dup   = p321.dup_pos
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class P321Wyckoff:
    """Container for Wyckoff data of space group P321 (No. 150).

    Attributes:
        spg_number: International number of the space group (150).
        wyckoff_table: Mapping from Wyckoff ID to (mul, add) tensors.
            ``mul`` has shape (k, 3, 3), ``add`` has shape (k, 3).
        wyckoff_site: List of Wyckoff labels ordered by ID
            (e.g., ["1a", "1b", "2c", ...]).
        num_equivalent_sites: List of multiplicities for each Wyckoff ID.
        dup_pos: Tensor of centering translations of shape (D, 3).
            For P321 (P-lattice), this is just ``[[0, 0, 0]]``.
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
    def _ops_6g() -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 6 general-position operators for P321 (Wyckoff 6g).

        Coordinate forms in fractional (x, y, z)::

            (x,       y,      z)
            (-y,      x-y,    z)
            (-x + y,  -x,     z)
            (y,       x,     -z)
            (x-y,     -y,    -z)
            (-x,      -x + y,-z)

        Each operator is represented as a 3x3 matrix ``M`` and a translation
        vector ``t`` such that ``p' = M @ p + t``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Ms: Tensor of shape (6, 3, 3), linear-part matrices.
                - Ts: Tensor of shape (6, 3), translation vectors
                  (all zeros for 6g).
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

        # (-y, x-y, z)
        Ms.append(
            torch.tensor(
                [[0.0, -1.0, 0.0],
                 [1.0, -1.0, 0.0],
                 [0.0,  0.0, 1.0]],
                dtype=torch.float32,
            )
        )

        # (-x + y, -x, z)
        Ms.append(
            torch.tensor(
                [[-1.0, 1.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        )

        # (y, x, -z)
        Ms.append(
            torch.tensor(
                [[0.0, 1.0,  0.0],
                 [1.0, 0.0,  0.0],
                 [0.0, 0.0, -1.0]],
                dtype=torch.float32,
            )
        )

        # (x-y, -y, -z)
        Ms.append(
            torch.tensor(
                [[1.0, -1.0,  0.0],
                 [0.0, -1.0,  0.0],
                 [0.0,  0.0, -1.0]],
                dtype=torch.float32,
            )
        )

        # (-x, -x + y, -z)
        Ms.append(
            torch.tensor(
                [[-1.0, 0.0,  0.0],
                 [-1.0, 1.0,  0.0],
                 [ 0.0, 0.0, -1.0]],
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
    def build(cls) -> "P321Wyckoff":
        """Build a :class:`P321Wyckoff` instance with all data populated.

        This method defines the affine operators for each Wyckoff position
        of P321 (1a–6g) in a straightforward, explicit way. No automatic
        deduplication logic is used.

        Returns:
            P321Wyckoff: Fully constructed Wyckoff data for P321.
        """
        # 1a : (0, 0, 0)   multiplicity 1
        M_a = torch.zeros((1, 3, 3), dtype=torch.float32)
        t_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        # 1b : (0, 0, 1/2)   multiplicity 1
        M_b = torch.zeros((1, 3, 3), dtype=torch.float32)
        t_b = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)

        # 2c : (0, 0, z), (0, 0, -z)   multiplicity 2
        # Only the z-component is used; x, y are ignored.
        M_c = torch.tensor(
            [
                # (0, 0,  z)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0]],
                # (0, 0, -z)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0,-1.0]],
            ],
            dtype=torch.float32,
        )
        t_c = torch.zeros((2, 3), dtype=torch.float32)

        # 2d : (1/3, 2/3, z), (2/3, 1/3, -z)   multiplicity 2
        M_d = M_c.clone()
        t_d = torch.tensor(
            [
                [1.0 / 3.0, 2.0 / 3.0, 0.0],  # +z branch
                [2.0 / 3.0, 1.0 / 3.0, 0.0],  # -z branch
            ],
            dtype=torch.float32,
        )

        # 3e : (x, 0, 0), (0, x, 0), (-x, -x, 0)   multiplicity 3
        # Only the x-parameter (first coordinate) is used.
        M_e = torch.tensor(
            [
                # (x, 0, 0)
                [[ 1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0]],
                # (0, x, 0)
                [[ 0.0, 0.0, 0.0],
                 [ 1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0]],
                # (-x, -x, 0)
                [[-1.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        t_e = torch.zeros((3, 3), dtype=torch.float32)

        # 3f : (x, 0, 1/2), (0, x, 1/2), (-x, -x, 1/2)   multiplicity 3
        M_f = M_e.clone()
        t_f = torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )

        # 6g : general position (x, y, z) with 6 operations
        M_g, t_g = cls._ops_6g()

        wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
            0: (M_a, t_a),  # 1a
            1: (M_b, t_b),  # 1b
            2: (M_c, t_c),  # 2c
            3: (M_d, t_d),  # 2d
            4: (M_e, t_e),  # 3e
            5: (M_f, t_f),  # 3f
            6: (M_g, t_g),  # 6g
        }

        wyckoff_site: List[str] = ["1a", "1b", "2c", "2d", "3e", "3f", "6g"]

        # P-lattice: only identity centering.
        dup_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        num_equiv = [
            wyckoff_table[i][0].shape[0] * dup_pos.shape[0]
            for i in range(len(wyckoff_table))
        ]

        return cls(
            spg_number=150,
            wyckoff_table=wyckoff_table,
            wyckoff_site=wyckoff_site,
            num_equivalent_sites=num_equiv,
            dup_pos=dup_pos,
        )
