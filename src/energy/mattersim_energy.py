from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Sequence

import torch

from src.utils.common import get_master_dir

if TYPE_CHECKING:
    from ase import Atoms


SMOACS_NUM_ELEMENTS = 98


def convert_smoacs_atom_types(
    atom_types: torch.Tensor, max_atomic_number: int
) -> torch.Tensor:
    """Convert SMOACS element columns (Z-1) to MatterSim columns (Z)."""
    if atom_types.ndim != 2 or atom_types.shape[1] != SMOACS_NUM_ELEMENTS:
        raise ValueError(
            "SMOACS atom_types must have shape (N, 98), "
            f"got {tuple(atom_types.shape)}"
        )
    if not 0 < max_atomic_number <= SMOACS_NUM_ELEMENTS:
        raise ValueError(
            "MatterSim max atomic number must be in [1, 98], "
            f"got {max_atomic_number}"
        )

    unsupported = atom_types[:, max_atomic_number:]
    if torch.count_nonzero(unsupported).item() != 0:
        raise ValueError(
            "SMOACS atom_types contains nonzero probability for elements "
            f"above MatterSim max_z={max_atomic_number}"
        )

    dummy_column = torch.zeros(
        (atom_types.shape[0], 1),
        device=atom_types.device,
        dtype=atom_types.dtype,
    )
    return torch.cat([dummy_column, atom_types[:, :max_atomic_number]], dim=1)


class MatterSimEnergyModule:
    """Compute differentiable MatterSim energies from SMOACS tensors."""

    def __init__(
        self,
        device: str,
        model_path: Optional[str] = None,
        twobody_cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        soft_normalize: bool = True,
        rebuild_graph_each_call: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model_path = model_path
        self.twobody_cutoff = twobody_cutoff
        self.threebody_cutoff = threebody_cutoff
        self.soft_normalize = soft_normalize
        self.rebuild_graph_each_call = rebuild_graph_each_call
        self._calculator = None
        self._max_z = None

    def compute_energy(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        sizes: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.compute_properties(
            atom_types, positions, lattice, sizes, batch_index
        )["total_energy"]

    def compute_properties(
        self,
        atom_types,
        positions,
        lattice,
        sizes,
        batch_index=None,
        include_forces=False,
        include_stresses=False,
        create_graph=False,
    ):
        """Use total energy for physical derivatives; copy evaluation inputs locally."""
        physical = include_forces or include_stresses
        if physical and not create_graph:
            with torch.inference_mode(False), torch.enable_grad():
                return {
                    key: value.detach()
                    for key, value in self._compute_properties(
                        atom_types.detach().clone(),
                        positions.detach().clone(),
                        lattice.detach().clone(),
                        sizes.detach().clone(),
                        None if batch_index is None else batch_index.detach().clone(),
                        include_forces,
                        include_stresses,
                        False,
                    ).items()
                }
        return self._compute_properties(
            atom_types,
            positions,
            lattice,
            sizes,
            batch_index,
            include_forces,
            include_stresses,
            create_graph,
        )

    def _compute_properties(
        self,
        atom_types,
        positions,
        lattice,
        sizes,
        batch_index,
        include_forces,
        include_stresses,
        create_graph,
    ):
        calc = self._load_calculator()
        if include_forces or include_stresses:
            calc.diff_model.eval()
            calc.diff_model.requires_grad_(False)

        atom_types = atom_types.to(self.device)
        positions = positions.to(self.device)
        lattice = lattice.to(self.device)
        sizes = sizes.to(self.device)

        if self._max_z is None:
            raise RuntimeError("MatterSim max_z is unavailable after model loading")
        atom_types = convert_smoacs_atom_types(
            atom_types, max_atomic_number=int(self._max_z)
        )
        atoms_list = self._build_atoms_list(
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            sizes=sizes,
        )

        if batch_index is None:
            batch_index = torch.repeat_interleave(
                torch.arange(len(sizes), device=self.device), sizes
            )

        output = calc.forward_batch(
            atoms_list=atoms_list,
            atom_types=atom_types,
            positions=positions,
            lattice=lattice,
            batch_index=batch_index,
            sizes=sizes,
            include_forces=include_forces,
            include_stresses=include_stresses,
            create_graph_forces=include_forces and create_graph,
            create_graph_stress=include_stresses and create_graph,
            soft_normalize=self.soft_normalize,
        )
        return output

    def _load_calculator(self):
        if self._calculator is None:
            try:
                from mattersim.forcefield.differentiable_potential import (
                    DifferentiableMatterSimCalculator,
                )
                from mattersim.forcefield.potential import Potential
            except Exception as exc:
                raise RuntimeError(
                    "MatterSim is not installed. Run: pip install -e ./differentiable_MatterSim"
                ) from exc

            model_path = self._resolve_model_path()
            try:
                if model_path is None:
                    potential = Potential.from_checkpoint(device=str(self.device))
                else:
                    potential = Potential.from_checkpoint(
                        load_path=model_path, device=str(self.device)
                    )
            except Exception as exc:
                raise RuntimeError(
                    "MatterSim model load failed. Ensure the submodule is initialized and "
                    "pass --mattersim_model if needed."
                ) from exc

            self._calculator = DifferentiableMatterSimCalculator(
                potential=potential,
                device=str(self.device),
                twobody_cutoff=self.twobody_cutoff,
                threebody_cutoff=self.threebody_cutoff,
            )
            self._max_z = self._calculator.diff_model.max_z

        if self.rebuild_graph_each_call:
            self._calculator.graph_builder_batch = None
        return self._calculator

    def _resolve_model_path(self) -> Optional[str]:
        if self.model_path:
            return self.model_path
        default_path = os.path.join(
            get_master_dir(),
            "differentiable_MatterSim",
            "pretrained_models",
            "mattersim-v1.0.0-1M.pth",
        )
        if os.path.exists(default_path):
            return default_path
        return None

    @property
    def max_atomic_number(self) -> int:
        self._load_calculator()
        if self._max_z is None:
            raise RuntimeError("MatterSim max_z is unavailable after model loading")
        return int(self._max_z)

    def _build_atoms_list(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        sizes: torch.Tensor,
    ) -> Sequence["Atoms"]:
        try:
            from ase import Atoms
        except Exception as exc:
            raise RuntimeError(
                "ASE is required for MatterSim topology building. "
                "Install via the differentiable_MatterSim dependencies."
            ) from exc

        sizes_list = sizes.detach().cpu().tolist()
        atomic_numbers = torch.argmax(atom_types, dim=1).detach().cpu().numpy()
        positions_np = positions.detach().cpu().numpy()
        lattice_np = lattice.detach().cpu().numpy()

        atoms_list = []
        offset = 0
        for idx, size in enumerate(sizes_list):
            atoms = Atoms(
                numbers=atomic_numbers[offset : offset + size],
                positions=positions_np[offset : offset + size],
                cell=lattice_np[idx],
                pbc=True,
            )
            atoms_list.append(atoms)
            offset += size
        return atoms_list
