# src/spg_wp/Fbar43m.py
"""Wyckoff positions for space group F-43m (No. 216, cubic axes).

This module provides a class-based container for the Wyckoff data of
F-43m.  Each Wyckoff position is defined by an explicit set of affine
operators (mul, add) such that

    r' = M @ r + t

with ``M`` of shape (3, 3) and ``t`` of shape (3,).

Typical usage
-------------

.. code-block:: python

    from Fbar43m import Fbar43mWyckoff

    f216 = Fbar43mWyckoff.build()
    table = f216.wyckoff_table
    dup   = f216.dup_pos

The Wyckoff letters and multiplicities follow the conventional setting
for F-43m (No. 216):

    4a, 4b, 4c, 4d, 16e, 24f, 24g, 48h, 96i
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class Fbar43mWyckoff:
    """Container for Wyckoff data of space group F-43m (No. 216).

    Attributes:
        spg_number:
            International number of the space group (216).
        wyckoff_table:
            Mapping from Wyckoff ID to (mul, add) tensors.
            ``mul`` has shape (k, 3, 3), ``add`` has shape (k, 3).
        wyckoff_site:
            List of Wyckoff labels ordered by ID
            (["4a", "4b", ..., "96i"]).
        num_equivalent_sites:
            List of multiplicities for each Wyckoff ID.
        dup_pos:
            Tensor of centering translations of shape (D, 3).
            For F-43m (F-lattice) this is the usual FCC set
            ``[(0,0,0), (0,1/2,1/2), (1/2,0,1/2), (1/2,1/2,0)]``.
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
    def _dup_fcc() -> torch.Tensor:
        """Return FCC centering translations for F-43m.

        Forms (in fractional coordinates)::

            (0,   0,   0)
            (0,   1/2, 1/2)
            (1/2, 0,   1/2)
            (1/2, 1/2, 0)

        Shape: (4, 3).
        """
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=torch.float32,
        )

    # ---- 96i : general position -------------------------------------- #

    @staticmethod
    def _ops_96i() -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 24 general-position operators for F-43m (Wyckoff 96i).

        The International Tables list the 24 affine linear parts (before
        FCC centering) as all permutations of (x, y, z) with an even
        number of sign flips.  In coordinates this is the set

        (x,  y,  z),  (-x, -y,  z),  (-x,  y, -z),  (x, -y, -z),
        (z,  x,  y),  (z, -x, -y),  (-z, -x,  y),  (-z,  x, -y),
        (y,  z,  x),  (-y,  z, -x), (y, -z, -x),  (-y, -z,  x),
        (y,  x,  z),  (-y, -x,  z), (y, -x, -z),  (-y,  x, -z),
        (x,  z,  y),  (-x,  z, -y), (-x, -z,  y), (x, -z, -y),
        (z,  y,  x),  (z, -y, -x),  (-z,  y, -x), (-z, -y,  x).

        Each of these is a pure rotation, i.e. translation-part t = 0.

        Returns:
            Ms: (24, 3, 3) tensor of rotation matrices.
            Ts: (24, 3) tensor of zeros.
        """
        Ms: List[torch.Tensor] = []
        Ts: List[torch.Tensor] = []

        # permutations of (x, y, z) encoded as index tuples
        perms = [
            (0, 1, 2),  # (x, y, z)
            (2, 0, 1),  # (z, x, y)
            (1, 2, 0),  # (y, z, x)
            (1, 0, 2),  # (y, x, z)
            (0, 2, 1),  # (x, z, y)
            (2, 1, 0),  # (z, y, x)
        ]

        # sign patterns with an even number of minus signs
        sign_patterns = [
            (1.0, 1.0, 1.0),   # + + +
            (-1.0, -1.0, 1.0),  # - - +
            (-1.0, 1.0, -1.0),  # - + -
            (1.0, -1.0, -1.0),  # + - -
        ]

        for p in perms:
            for s in sign_patterns:
                M = torch.zeros((3, 3), dtype=torch.float32)
                # each output coordinate picks one input coordinate with a sign
                for row, (axis, sign) in enumerate(zip(p, s)):
                    M[row, axis] = sign
                Ms.append(M)
                Ts.append(torch.zeros(3, dtype=torch.float32))

        return torch.stack(Ms, dim=0), torch.stack(Ts, dim=0)

    # ---- 48h : x,x,z-type special position --------------------------- #

    @staticmethod
    def _ops_48h() -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 12 operators for Wyckoff 48h (..m).

        Coordinate forms (before FCC centering)::

            (x,  x,  z)
            (-x, -x, z)
            (-x, x, -z)
            (x,  -x, -z)

            (z,  x,  x)
            (z,  -x, -x)
            (-z, -x, x)
            (-z, x,  -x)

            (x,  z,  x)
            (-x, z,  -x)
            (x,  -z, -x)
            (-x, -z, x)

        These depend only on parameters x (first coordinate) and z
        (third coordinate) of ``r = (x, y, z)``.
        """
        Ms: List[torch.Tensor] = []
        Ts: List[torch.Tensor] = []

        # Prototype patterns ("x" or "z" in each coordinate)
        proto_syms = [
            ("x", "x", "z"),
            ("x", "z", "x"),
            ("z", "x", "x"),
        ]
        symbol_to_col = {"x": 0, "z": 2}

        # Same even-minus sign patterns as for 96i
        sign_patterns = [
            (1.0, 1.0, 1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
        ]

        for proto in proto_syms:
            cols = [symbol_to_col[s] for s in proto]
            for sx, sy, sz in sign_patterns:
                M = torch.zeros((3, 3), dtype=torch.float32)
                for row, (col, sign) in enumerate(zip(cols, (sx, sy, sz))):
                    M[row, col] = sign
                Ms.append(M)
                Ts.append(torch.zeros(3, dtype=torch.float32))

        return torch.stack(Ms, dim=0), torch.stack(Ts, dim=0)

    # ------------------------------------------------------------------ #
    # Constructor                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def build(cls) -> "Fbar43mWyckoff":
        """Build a :class:`Fbar43mWyckoff` instance with all data populated.

        The affine operators for each Wyckoff position (4a–96i) are
        defined explicitly in terms of simple linear maps and fractional
        translations.  FCC centering is handled via :attr:`dup_pos`.

        Returns:
            Fbar43mWyckoff: Fully constructed Wyckoff data for F-43m.
        """
        # -------------------------- 4a–4d ------------------------------ #
        # Each of these has one representative per FCC sublattice.
        # Multiplicity = 1 * 4 (dup_pos) = 4.

        # 4a : (0, 0, 0)
        M_a = torch.zeros((1, 3, 3), dtype=torch.float32)
        t_a = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

        # 4b : (1/2, 1/2, 1/2)
        M_b = torch.zeros((1, 3, 3), dtype=torch.float32)
        t_b = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)

        # 4c : (1/4, 1/4, 1/4)
        M_c = torch.zeros((1, 3, 3), dtype=torch.float32)
        t_c = torch.tensor([[0.25, 0.25, 0.25]], dtype=torch.float32)

        # 4d : (3/4, 3/4, 3/4)
        M_d = torch.zeros((1, 3, 3), dtype=torch.float32)
        t_d = torch.tensor([[0.75, 0.75, 0.75]], dtype=torch.float32)

        # -------------------------- 16e ------------------------------- #
        # 16e : .3m
        # Coordinate forms (before centering)::
        #
        #   (x,  x,  x)
        #   (-x, -x, x)
        #   (-x, x,  -x)
        #   (x,  -x, -x)
        #
        # Only the first input parameter "x" (r[0]) is used.

        M_e = torch.tensor(
            [
                # (x, x, x)
                [[1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0]],
                # (-x, -x, x)
                [[-1.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 1.0, 0.0, 0.0]],
                # (-x, x, -x)
                [[-1.0, 0.0, 0.0],
                 [ 1.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0]],
                # (x, -x, -x)
                [[ 1.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        t_e = torch.zeros((4, 3), dtype=torch.float32)

        # -------------------------- 24f ------------------------------- #
        # 24f : 2.mm
        #
        # Coordinate forms (before centering)::
        #
        #   (x,   0,   0)
        #   (-x,  0,   0)
        #   (0,   x,   0)
        #   (0,  -x,   0)
        #   (0,   0,   x)
        #   (0,   0,  -x)
        #
        # Again only the first parameter "x" is used.

        M_f = torch.tensor(
            [
                # (x, 0, 0)
                [[1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]],
                # (-x, 0, 0)
                [[-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0]],
                # (0, x, 0)
                [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0]],
                # (0, -x, 0)
                [[0.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0],
                 [ 0.0, 0.0, 0.0]],
                # (0, 0, x)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0]],
                # (0, 0, -x)
                [[0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [-1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        t_f = torch.zeros((6, 3), dtype=torch.float32)

        # -------------------------- 24g ------------------------------- #
        # 24g : 2.mm
        #
        # Coordinate forms (before centering)::
        #
        #   (x,       1/4,     1/4)
        #   (-x,      3/4,     1/4)
        #   (1/4,     x,       1/4)
        #   (1/4,     -x,      3/4)
        #   (1/4,     1/4,     x)
        #   (3/4,     1/4,     -x)
        #
        # Linear dependence on the parameter is identical to 24f;
        # only the translations differ.

        M_g = M_f.clone()
        t_g = torch.tensor(
            [
                [0.0, 0.25, 0.25],  # (x, 1/4, 1/4)
                [0.0, 0.75, 0.25],  # (-x, 3/4, 1/4)
                [0.25, 0.0, 0.25],  # (1/4, x, 1/4)
                [0.25, 0.0, 0.75],  # (1/4, -x, 3/4)
                [0.25, 0.25, 0.0],  # (1/4, 1/4, x)
                [0.75, 0.25, 0.0],  # (3/4, 1/4, -x)
            ],
            dtype=torch.float32,
        )

        # -------------------------- 48h ------------------------------- #
        M_h, t_h = cls._ops_48h()

        # -------------------------- 96i ------------------------------- #
        M_i, t_i = cls._ops_96i()

        # ------------------------------------------------------------------
        # Assemble table and metadata
        # ------------------------------------------------------------------
        wyckoff_table: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
            0: (M_a, t_a),  # 4a
            1: (M_b, t_b),  # 4b
            2: (M_c, t_c),  # 4c
            3: (M_d, t_d),  # 4d
            4: (M_e, t_e),  # 16e
            5: (M_f, t_f),  # 24f
            6: (M_g, t_g),  # 24g
            7: (M_h, t_h),  # 48h
            8: (M_i, t_i),  # 96i
        }

        wyckoff_site: List[str] = [
            "4a",
            "4b",
            "4c",
            "4d",
            "16e",
            "24f",
            "24g",
            "48h",
            "96i",
        ]

        # FCC lattice centering
        dup_pos = cls._dup_fcc()

        num_equiv = [
            wyckoff_table[i][0].shape[0] * dup_pos.shape[0]
            for i in range(len(wyckoff_table))
        ]

        return cls(
            spg_number=216,
            wyckoff_table=wyckoff_table,
            wyckoff_site=wyckoff_site,
            num_equivalent_sites=num_equiv,
            dup_pos=dup_pos,
        )

