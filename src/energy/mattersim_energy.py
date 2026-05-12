from __future__ import annotations

from typing import Optional, Sequence

import os

import torch

from src.utils.common import get_master_dir


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
        calc = self._load_calculator()

        atom_types = atom_types.to(self.device)
        positions = positions.to(self.device)
        lattice = lattice.to(self.device)
        sizes = sizes.to(self.device)

        atom_types = self._normalize_atom_types(atom_types)
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
            include_forces=False,
            include_stresses=False,
            soft_normalize=self.soft_normalize,
        )
        return output["total_energy"]

    def _load_calculator(self):
        if self._calculator is None:
            try:
                from mattersim.forcefield.differentiable_potential import (
                    DifferentiableMatterSimCalculator,
                )
                from mattersim.forcefield.potential import Potential
            except Exception as exc:
                raise RuntimeError(
                    "MatterSim is not installed. Run: pip install -e ./differentiable-mattersim"
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
            "differentiable-mattersim",
            "pretrained_models",
            "mattersim-v1.0.0-1M.pth",
        )
        if os.path.exists(default_path):
            return default_path
        return None

    def _normalize_atom_types(self, atom_types: torch.Tensor) -> torch.Tensor:
        if self._max_z is None:
            return atom_types
        target_dim = self._max_z + 1
        current_dim = atom_types.shape[1]
        if current_dim == target_dim:
            return atom_types
        if current_dim > target_dim:
            extra = atom_types[:, target_dim:]
            if torch.any(extra > 0):
                print(
                    "<info> MatterSim max_z is smaller than atom_types. "
                    "Truncating higher-Z entries for energy calculation."
                )
                truncated = atom_types[:, :target_dim]
                row_sum = truncated.sum(dim=1, keepdim=True)
                return torch.where(row_sum > 0, truncated / row_sum, truncated)
            return atom_types[:, :target_dim]
        pad_size = target_dim - current_dim
        padding = torch.zeros(
            (atom_types.shape[0], pad_size),
            device=atom_types.device,
            dtype=atom_types.dtype,
        )
        return torch.cat([atom_types, padding], dim=1)

    def _build_atoms_list(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        lattice: torch.Tensor,
        sizes: torch.Tensor,
    ) -> Sequence["ase.Atoms"]:
        try:
            from ase import Atoms
        except Exception as exc:
            raise RuntimeError(
                "ASE is required for MatterSim topology building. "
                "Install via the differentiable-mattersim dependencies."
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
