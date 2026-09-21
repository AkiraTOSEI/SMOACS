from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.energy import MatterSimEnergyModule
from src.energy.mattersim_energy import convert_smoacs_atom_types
from src.preprocess.alignn_preprocess import ALIGNN_preprocess
from src.preprocess.crystalformer_preprocess import Crystalformer_preprocess
from src.preprocess.prepro_utils import apply_element_mask, create_element_mask
from src.utils.feature_utils import calculate_onehot
from src.utils.scalers import ABC_Scaler, Angle_Scaler


def _model_path() -> Path:
    configured_path = os.environ.get("MATTERSIM_TEST_MODEL")
    if configured_path is not None:
        return Path(configured_path)
    return (
        Path.home() / ".local/mattersim/pretrained_models" / "mattersim-v1.0.0-5M.pth"
    )


def test_smoacs_element_columns_are_shifted_to_atomic_numbers():
    atomic_numbers = torch.tensor([1, 8, 22, 56], dtype=torch.long)
    smoacs_atom_types = F.one_hot(atomic_numbers - 1, num_classes=98).float()
    smoacs_atom_types.requires_grad_(True)

    actual = convert_smoacs_atom_types(smoacs_atom_types, max_atomic_number=94)
    expected = F.one_hot(atomic_numbers, num_classes=95).float()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.argmax(actual, dim=1).tolist() == [1, 8, 22, 56]
    assert torch.count_nonzero(actual[:, 0]).item() == 0
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(4), rtol=0, atol=0)

    actual.sum().backward()
    assert smoacs_atom_types.grad is not None
    assert torch.count_nonzero(smoacs_atom_types.grad[:, :94]).item() > 0


def test_all_supported_columns_and_soft_interpolation():
    all_elements = torch.eye(98)[:94].requires_grad_()
    actual = convert_smoacs_atom_types(all_elements, 94)
    torch.testing.assert_close(actual, F.one_hot(torch.arange(1, 95), 95).float())
    soft = 0.3 * all_elements[0:1] + 0.7 * all_elements[93:94]
    converted = convert_smoacs_atom_types(soft, 94)
    assert converted[0, 1].item() == pytest.approx(0.3)
    assert converted[0, 94].item() == pytest.approx(0.7)
    converted[:, 94].sum().backward()
    assert all_elements.grad[93, 93].item() == pytest.approx(0.7)


def test_conversion_rejects_unsupported_atomic_numbers():
    atom_types = torch.zeros((1, 98))
    atom_types[0, 94] = 1

    with pytest.raises(ValueError, match="above MatterSim max_z=94"):
        convert_smoacs_atom_types(atom_types, max_atomic_number=94)


class _RecordingCalculator:
    def __init__(self):
        self.graph_builder_batch = None
        self.atoms_list = None
        self.atom_types = None

    def forward_batch(self, *, atoms_list, atom_types, **kwargs):
        self.atoms_list = atoms_list
        self.atom_types = atom_types
        return {"total_energy": torch.zeros(len(atoms_list))}


def test_compute_energy_uses_shifted_columns_for_atoms_and_model_input():
    atomic_numbers = torch.tensor([1, 8, 22, 56], dtype=torch.long)
    smoacs_atom_types = F.one_hot(atomic_numbers - 1, num_classes=98).float()
    recorder = _RecordingCalculator()
    module = MatterSimEnergyModule(device="cpu")
    module._calculator = recorder
    module._max_z = 94

    module.compute_energy(
        atom_types=smoacs_atom_types,
        positions=torch.zeros((4, 3)),
        lattice=torch.eye(3).mul(10).unsqueeze(0),
        sizes=torch.tensor([4]),
    )

    assert recorder.atom_types is not None
    assert torch.argmax(recorder.atom_types, dim=1).tolist() == [1, 8, 22, 56]
    assert recorder.atoms_list is not None
    assert recorder.atoms_list[0].get_atomic_numbers().tolist() == [1, 8, 22, 56]


def test_element_mask_excludes_atomic_numbers_95_through_98():
    distribution = torch.ones((2, 98), requires_grad=True)
    element_mask = create_element_mask(
        num_elements=98,
        max_atomic_number=94,
        device=distribution.device,
        dtype=distribution.dtype,
    )

    masked = apply_element_mask(distribution, element_mask)

    assert element_mask[93].item() == 1
    assert torch.count_nonzero(element_mask[94:]).item() == 0
    assert torch.count_nonzero(masked[:, 94:]).item() == 0
    torch.testing.assert_close(masked.sum(dim=1), torch.ones(2), rtol=0, atol=1e-7)

    masked[0, 0].backward()
    assert distribution.grad is not None
    assert torch.count_nonzero(distribution.grad[0, :94]).item() > 0
    assert torch.count_nonzero(distribution.grad[:, 94:]).item() == 0


def test_element_mask_rejects_rows_without_supported_elements():
    distribution = torch.zeros((1, 98))
    distribution[0, 94] = 1
    element_mask = create_element_mask(
        num_elements=98,
        max_atomic_number=94,
        device=distribution.device,
        dtype=distribution.dtype,
    )

    with pytest.raises(ValueError, match="No MatterSim-supported element"):
        apply_element_mask(distribution, element_mask)


def test_final_onehot_cannot_select_an_unsupported_element():
    element_mask = create_element_mask(
        num_elements=98,
        max_atomic_number=94,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    atomic_distribution = torch.zeros((1, 98))
    atomic_distribution[0, 93] = 1
    atomic_distribution[0, 97] = 10
    atomic_mask = torch.ones((1, 98, 1)) * element_mask.view(1, 98, 1)
    normalized_dist = apply_element_mask(
        torch.softmax(atomic_distribution, dim=1), element_mask
    )

    _, onehot_x, _, _ = calculate_onehot(
        {
            "normalized_dist": normalized_dist,
            "sharpened_ox_mask": torch.ones((1, 1)),
            "atomic_distribution": atomic_distribution,
            "atomic_mask": atomic_mask,
        }
    )

    assert torch.argmax(onehot_x, dim=1).item() == 93
    assert torch.count_nonzero(onehot_x[:, 94:]).item() == 0


def test_alignn_preprocess_outputs_exact_zero_for_unsupported_elements():
    initial_abc = torch.tensor([[5.0, 5.0, 5.0]])
    initial_angles = torch.tensor([[90.0, 90.0, 90.0]])
    abc_scaler = ABC_Scaler(initial_abc, 2.0, 10.0, device="cpu")
    angle_scaler = Angle_Scaler(30.0, 150.0)
    atomic_distribution = torch.zeros((1, 98))
    atomic_distribution[0, 0] = 1
    atomic_distribution[0, 97] = 10
    element_mask = create_element_mask(
        num_elements=98,
        max_atomic_number=94,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    outputs = ALIGNN_preprocess(
        scaled_batch_abc=abc_scaler.scale(initial_abc),
        scaled_batch_angle=angle_scaler.scale(initial_angles),
        batch_dir_coords=torch.zeros((1, 3)),
        atomic_distribution=atomic_distribution,
        batch_dst_ids=torch.tensor([0]),
        batch_src_ids=torch.tensor([0]),
        batch_displace=torch.tensor([[1.0, 0.0, 0.0]]),
        z_src_ids=torch.tensor([0]),
        z_dst_ids=torch.tensor([0]),
        size=torch.tensor([1]),
        wp_size=torch.tensor([1]),
        site_ids=torch.tensor([0]),
        init_coords=torch.zeros((1, 3)),
        temperature=0.01,
        abc_scaler=abc_scaler,
        angle_scaler=angle_scaler,
        num_edges=torch.tensor([1]),
        n_nodes=1,
        atom_feat_matrix=torch.eye(98),
        limit_coords_displacement=None,
        adding_noise_scale=None,
        abc_range=(2.0, 10.0),
        angle_range=(30.0, 150.0),
        atomic_mask=torch.ones((1, 98, 1)),
        ox_states_used_mask=torch.ones((1, 1)),
        ox_mask_learnable_tensor_per_crystal=torch.ones((1, 1)),
        element_mask=element_mask,
        atomic_dist_Z_more_than_86=False,
    )
    normalized_dist = outputs[6]

    assert torch.count_nonzero(normalized_dist[:, 94:]).item() == 0
    torch.testing.assert_close(
        normalized_dist.sum(dim=1), torch.ones(1), rtol=0, atol=1e-7
    )


def test_crystalformer_preprocess_outputs_exact_zero_for_unsupported_elements():
    initial_abc = torch.tensor([[5.0, 5.0, 5.0]])
    initial_angles = torch.tensor([[90.0, 90.0, 90.0]])
    abc_scaler = ABC_Scaler(initial_abc, 2.0, 10.0, device="cpu")
    angle_scaler = Angle_Scaler(30.0, 150.0)
    atomic_distribution = torch.zeros((1, 98))
    atomic_distribution[0, 0] = 1
    atomic_distribution[0, 97] = 10
    element_mask = create_element_mask(
        num_elements=98,
        max_atomic_number=94,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    outputs = Crystalformer_preprocess(
        scaled_batch_abc=abc_scaler.scale(initial_abc),
        scaled_batch_angle=angle_scaler.scale(initial_angles),
        batch_dir_coords=torch.zeros((1, 3)),
        atomic_distribution=atomic_distribution,
        init_coords=torch.zeros((1, 3)),
        size=torch.tensor([1]),
        temperature=0.01,
        abc_scaler=abc_scaler,
        angle_scaler=angle_scaler,
        atomic_mask=torch.ones((1, 98, 1)),
        ox_states_used_mask=torch.ones((1, 1)),
        limit_coords_displacement=None,
        ox_mask_learnable_tensor_per_crystal=torch.ones((1, 1)),
        adding_noise_scale=None,
        abc_range=(2.0, 10.0),
        angle_range=(30.0, 150.0),
        element_mask=element_mask,
        atomic_dist_Z_more_than_86=False,
    )
    normalized_dist = outputs[3]

    assert torch.count_nonzero(normalized_dist[:, 94:]).item() == 0
    torch.testing.assert_close(
        normalized_dist.sum(dim=1), torch.ones(1), rtol=0, atol=1e-7
    )


def test_smoacs_energy_matches_native_mattersim_for_poscar():
    pytest.importorskip("mattersim")
    from ase.io import read
    from mattersim.forcefield import MatterSimCalculator
    from mattersim.forcefield.potential import Potential

    model_path = _model_path()
    if not model_path.is_file():
        pytest.skip("MatterSim model checkpoint not found")

    poscar_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "mattersim" / "BaTiO3.vasp"
    )
    atoms = read(poscar_path, format="vasp")

    potential = Potential.from_checkpoint(load_path=str(model_path), device="cpu")
    atoms.calc = MatterSimCalculator(
        potential=potential,
        compute_stress=False,
        device="cpu",
    )
    expected_energy = torch.tensor([atoms.get_potential_energy()], dtype=torch.float32)

    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    smoacs_atom_types = F.one_hot(atomic_numbers - 1, num_classes=98).float()
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    lattice = torch.tensor(atoms.cell.array, dtype=torch.float32).unsqueeze(0)
    sizes = torch.tensor([len(atoms)], dtype=torch.long)

    module = MatterSimEnergyModule(device="cpu", model_path=str(model_path))
    actual_energy = module.compute_energy(
        atom_types=smoacs_atom_types,
        positions=positions,
        lattice=lattice,
        sizes=sizes,
    ).detach()

    torch.testing.assert_close(
        actual_energy,
        expected_energy,
        rtol=1e-6,
        atol=1e-5,
    )
