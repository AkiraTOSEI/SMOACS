from typing import List, Tuple
from jarvis.core.atoms import Atoms
from typing import Any, Sequence
import torch
from string import ascii_lowercase
import re
LETTER_MAP = {ch: i for i, ch in enumerate(ascii_lowercase)}


def wyckoff_letter_index(tag: str) -> int:
    """
    Convert a Wyckoff tag like ``"1a"`` or ``"4e"`` to its 0-based
    alphabetical index (a→0, b→1, …, z→25).

    Parameters
    ----------
    tag : str
        A Wyckoff label of the form ``"<digits><letter>"``.
        The letter **must be exactly one character**.

    Returns
    -------
    int
        The alphabetical index of the Wyckoff letter.

    Raises
    ------
    ValueError
        If the tag does not match the required pattern or
        contains more than one alphabetical character.

    Examples
    --------
    >>> wyckoff_letter_index("1a")
    0
    >>> wyckoff_letter_index("4e")
    4
    >>> wyckoff_letter_index("5et")
    Traceback (most recent call last):
        ...
    ValueError: Invalid Wyckoff tag: '5et'
    """
    # strict pattern: 1 + digits, then exactly **one** letter a–z
    m = re.fullmatch(r"\d+([a-zA-Z])", tag)
    if m is None:
        raise ValueError(f"Invalid Wyckoff tag: {tag!r}")

    letter = m.group(1).lower()
    
    return LETTER_MAP[letter]

class WyckoffExpander:
    """
    Class to handle Wyckoff expansion for structures with symmetry operations.
    """

    def __init__(
            self, 
            WYCKOFF_TABLE: dict, 
            dup_positions: torch.Tensor,
            wyckoff_site: List[str],
            spg_number:int, 
            spg_name: str,
            num_equivalent_sites:List[int],
            device='cuda'
            ):
        """
        Initialize the WyckoffExpander with a Wyckoff table and centering translations.

        Args:
            WYCKOFF_TABLE (dict):  
                Dictionary mapping Wyckoff IDs to their symmetry operations.
            dup_positions (torch.Tensor):  
                Centering translations of shape ``(D, 3)``.
        """
        self.WYCKOFF_TABLE = WYCKOFF_TABLE
        self.dup_positions = dup_positions
        self.wyckoff_site = wyckoff_site
        self.spg_number = spg_number
        self.num_equivalent_sites = num_equivalent_sites
        self.spg_name = spg_name
        self.device = device

        # Ensure the WYCKOFF_TABLE is on the correct device
        if isinstance(self.WYCKOFF_TABLE, dict):
            self.WYCKOFF_TABLE_to_Device(self.WYCKOFF_TABLE)


    def WYCKOFF_TABLE_to_Device(self, WYCKOFF_TABLE):
        self.dup_positions = self.dup_positions.to(self.device)
        for key in WYCKOFF_TABLE.keys():
            WYCKOFF_TABLE[key] = (
                WYCKOFF_TABLE[key][0].to(self.device),
                WYCKOFF_TABLE[key][1].to(self.device)
            )

    def _to_matrix_ops(self, mult: torch.Tensor) -> torch.Tensor:
        """
        (k,3) -> (k,3,3) の対角行列化。すでに (k,3,3) ならそのまま返す。
        """
        if mult.ndim == 3:
            assert mult.shape[1:] == (3, 3)
            return mult
        elif mult.ndim == 2:
            assert mult.shape[1] == 3
            k = mult.shape[0]
            M = torch.zeros((k, 3, 3), dtype=mult.dtype, device=mult.device)
            M[:, 0, 0] = mult[:, 0]
            M[:, 1, 1] = mult[:, 1]
            M[:, 2, 2] = mult[:, 2]
            return M
        else:
            raise ValueError(f"invalid mult shape: {mult.shape}")
    
    def wyckoff_expand(
        self,
        wp_coords: torch.Tensor,
        wp_atom_dist: torch.Tensor,
        site_ids: torch.Tensor,
        return_site_ids: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Expand fractional coordinates and element-distribution vectors according to
        Wyckoff symmetry operations *and* centering translations.

        Args:
            wp_coords (torch.Tensor):  
                Fractional coordinates with shape ``(N, 3)`` and
                ``requires_grad`` if back-propagation is needed.
            wp_atom_dist (torch.Tensor):  
                Element-distribution (e.g. 98-dim one-hot/soft) with shape
                ``(N, F)`` where ``F`` is typically 98.
            site_ids (torch.Tensor):  
                Integer tensor of shape ``(N,)`` indicating the Wyckoff ID
                assigned to each site (must match keys in ``WYCKOFF_TABLE``).
            dup_positions (torch.Tensor):  
                Centering translations of shape ``(D, 3)``  
                *This argument must be provided; the function does not handle
                ``None``.*

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:  
                * **coords** – expanded, wrapped fractional coordinates  
                with shape ``(N_expanded, 3)``.  
                * **atom_dist** – expanded element-distribution vectors  
                with shape ``(N_expanded, F)``.  
                Here ``N_expanded = Σ_i (mult_i × D)``, i.e. original multiplicity
                times the number of centering vectors.
        """
        # -------- sanity checks --------------------------------------------------
        assert wp_coords.shape[0] == wp_atom_dist.shape[0], \
            "wp_coords and wp_atom_dist must have the same length"
        assert wp_coords.shape[1] == 3, "wp_coords must be of shape (N, 3). but got {}".format(wp_coords.shape)
        assert self.dup_positions is not None, "dup_positions must be provided"
        assert site_ids.shape[0] == wp_coords.shape[0] == wp_atom_dist.shape[0], \
            "site_ids must match wp_coords and wp_atom_dist in length"
        assert len(site_ids.shape) == 1, "site_ids must be a 1D tensor"
        assert len(wp_coords.shape) == 2, "wp_coords must be a 2D tensor"
        assert len(wp_atom_dist.shape) == 2, "wp_atom_dist must be a 2D tensor"

        # --- ① パディングを 0 で初期化 -----------------------
        W     = len(self.WYCKOFF_TABLE)
        Kmax  = max(self.num_equivalent_sites)

        mult_pad = torch.zeros((W, Kmax, 3, 3), device=self.device)
        add_pad  = torch.zeros((W, Kmax, 3),    device=self.device)
        valid_k  = torch.zeros((W, Kmax), dtype=torch.bool, device=self.device)

        for w_id, (mult, add) in self.WYCKOFF_TABLE.items():
            # (k,3) でも (k,3,3) でも受け付ける
            mult = self._to_matrix_ops(mult.to(self.device))
            add  = add.to(self.device)
            k = mult.shape[0]

            mult_pad[w_id, :k, :, :] = mult
            add_pad [w_id, :k, :]    = add
            valid_k [w_id, :k]       = True
            
        # --- ② ベクトル化展開 -------------------------------
        N, D = site_ids.shape[0], self.dup_positions.shape[0]
        F    = wp_atom_dist.shape[1]

        mult_row = mult_pad[site_ids]        # (N,Kmax,3)
        add_row  = add_pad [site_ids]        # (N,Kmax,3)
        mask_row = valid_k [site_ids]        # (N,Kmax)

        coords_wy = torch.einsum('nkij,nkj->nki', mult_row, wp_coords[:, None, :]) + add_row
        coords_wy = coords_wy * mask_row.unsqueeze(-1)           

        coords_cen = coords_wy[:,:,None,:] + self.dup_positions   # (N,Kmax,D,3)
        atom_wy    = wp_atom_dist[:,None,:].expand(-1,Kmax,-1)
        atom_wy    = atom_wy * mask_row.unsqueeze(-1)            
        atom_cen   = atom_wy[:,:,None,:].expand(-1,-1,D,-1)

        sid_cen = site_ids[:,None,None].expand(-1,Kmax,D)         # (N,Kmax,D)
        valid_mask = mask_row[:,:,None].expand(-1,-1,D)           # (N,Kmax,D)

        flat = lambda t: t.reshape(-1, *t.shape[3:])              # helper
        coords_out = flat(coords_cen)[valid_mask.reshape(-1)] % 1.0
        atom_out   = flat(atom_cen )[valid_mask.reshape(-1)]
        sid_out    = sid_cen.reshape(-1)[valid_mask.reshape(-1)]

        assert not torch.isnan(coords_out).any()
        assert not torch.isnan(atom_out).any()

        num_equiv_per_row = torch.tensor(self.num_equivalent_sites, device=site_ids.device)[site_ids]
        assert coords_out.shape[0] == atom_out.shape[0] == sid_out.shape[0] == torch.sum(num_equiv_per_row)
        
        assert torch.equal(
            torch.unique_consecutive(site_ids),
            torch.unique_consecutive(sid_out)
        )
        
        if return_site_ids:
            return coords_out, atom_out, sid_out
        else:
            return coords_out, atom_out





def assert_wyckoff_expansion_matches_atoms(
    atoms: Atoms,
    wp_coords: Sequence[Sequence[float]],
    site_ids: Sequence[int],
    wyckoff_expander: WyckoffExpander,
) -> None:
    """Assert that Wyckoff expansion reproduces the atomic fractional coordinates.

    Args:
        atoms: An object exposing a ``frac_coords`` attribute (e.g. a Pymatgen
            ``Structure``). The attribute must be convertible to a
            ``torch.Tensor``.
        wp_coords: Fractional Wyckoff position coordinates with shape
            ``(N_wp, 3)``.
        site_ids: Site identifiers corresponding one-to-one with ``wp_coords``.
        wyckoff_expander: An object providing
            ``wyckoff_expand(wp_coords, wp_atom_dist, site_ids)`` and returning
            a tuple where the first element is a tensor of expanded coordinates.

    Raises:
        AssertionError: If the coordinates returned by
            ``wyckoff_expander.wyckoff_expand`` differ from
            ``atoms.frac_coords`` beyond PyTorch’s default tolerance.
    """
    # Convert fractional coordinates from the atoms object to float32 tensor.
    frac_coords = torch.tensor(atoms.frac_coords, dtype=torch.float32)

    # Execute Wyckoff expansion with zero distance weights (keeps logic intact).
    expanded_coords, _ = wyckoff_expander.wyckoff_expand(
        wp_coords=torch.tensor(wp_coords).to(wyckoff_expander.device),
        wp_atom_dist=torch.zeros_like(torch.tensor(wp_coords)).to(wyckoff_expander.device),
        site_ids=torch.tensor(site_ids).to(wyckoff_expander.device),
    )

    # Verify that expanded coordinates match the original atomic coordinates.
    torch.testing.assert_close(expanded_coords, torch.tensor(frac_coords).to(wyckoff_expander.device))

