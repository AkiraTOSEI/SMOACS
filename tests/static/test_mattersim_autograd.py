"""Local 5M energy and mixed-derivative checks; no downloads or temporary files."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read
from ase.units import GPa

from src.energy.mattersim_energy import MatterSimEnergyModule
from src.losses.force_pressure_loss import (
    force_pressure_losses,
    weighted_force_pressure,
)
from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)


@pytest.fixture(scope="module")
def module():
    pytest.importorskip("mattersim")
    model = Path(
        os.environ.get(
            "MATTERSIM_TEST_MODEL",
            str(
                Path.home()
                / ".local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"
            ),
        )
    )
    if not model.is_file():
        pytest.skip("MatterSim 5M checkpoint not found")
    return MatterSimEnergyModule("cpu", model_path=str(model))


def sample():
    # Fixed two-site carbon structure from MP-3347313, distorted away from equilibrium.
    path = Path(__file__).parents[1] / "fixtures/mattersim/C2.vasp"
    atoms = read(path, format="vasp")
    rng = np.random.default_rng(918)
    atoms.positions += rng.normal(0, 0.025, atoms.positions.shape)
    atoms.set_cell(atoms.cell.array @ np.diag([1.02, 0.99, 1.01]), scale_atoms=True)
    return atoms


def inputs(atoms, dtype=torch.float32):
    return dict(
        atom_types=torch.nn.functional.one_hot(torch.tensor(atoms.numbers) - 1, 98).to(
            dtype
        ),
        positions=torch.tensor(atoms.positions, dtype=dtype),
        lattice=torch.tensor(atoms.cell.array, dtype=dtype)[None],
        sizes=torch.tensor([len(atoms)]),
    )


@pytest.mark.parametrize("term", ["force", "pressure"])
@pytest.mark.parametrize("mode", ["raw", "log"])
def test_mixed_derivatives_and_small_steps(module, term, mode):
    atoms = sample()
    initial = inputs(atoms)
    # Two allowed oxidation patterns modulate an unsaturated elemental mixture.
    atom_logits = torch.tensor([[0.2, -0.1], [-0.2, 0.3]], requires_grad=True)
    ox_logits = torch.tensor([0.1, -0.2], requires_grad=True)
    frac = torch.tensor(
        atoms.get_scaled_positions(), dtype=torch.float32, requires_grad=True
    )
    lattice = initial["lattice"].clone().requires_grad_()
    basis = torch.nn.functional.one_hot(torch.tensor([5, 6]), 98).float()
    config = dict(
        use_force_loss=term == "force",
        use_pressure_loss=term == "pressure",
        force_loss_mode=mode,
        pressure_loss_mode=mode,
        F_ref=1.0,
        P_ref=1.0,
        force_loss_clip=1e-7,
        pressure_loss_clip=1e-7,
    )

    def loss(variables, create_graph=True):
        atomic, oxidation, fractional, cell = variables
        weights, _ = masking_atomic_distribution(
            torch.nn.functional.softplus(atomic),
            torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]).expand(2, -1, -1),
            torch.ones(2, 2),
            stacking_learnable_oxsides_mask(
                torch.nn.functional.softplus(oxidation)[None], initial["sizes"]
            ),
            softmax_temp=0.5,
        )
        dist = weights @ basis
        output = module.compute_properties(
            atom_types=dist,
            positions=fractional @ cell[0],
            lattice=cell,
            sizes=initial["sizes"],
            include_forces=term == "force",
            include_stresses=term == "pressure",
            create_graph=create_graph,
        )
        return weighted_force_pressure(
            force_pressure_losses(output, initial["sizes"], config)
        ).mean()

    variables = [atom_logits, ox_logits, frac, lattice]
    value = loss(variables)
    gradients = torch.autograd.grad(value, variables)
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    assert all(
        p.grad is None and not p.requires_grad
        for p in module._calculator.diff_model.parameters()
    )
    for index, gradient in enumerate(gradients):
        direction = gradient.detach() / gradient.norm()
        analytic = (gradient * direction).sum().item()
        estimates = []
        for step in (3e-3, 1e-3, 3e-4):
            plus = [x.detach().clone() for x in variables]
            minus = [x.detach().clone() for x in variables]
            plus[index] += step * direction
            minus[index] -= step * direction
            finite = ((loss(plus, False) - loss(minus, False)) / (2 * step)).item()
            estimates.append(finite)
        print("directional", term, mode, index, analytic, estimates)
        assert (
            min(abs(x - analytic) for x in estimates) <= 0.005 * abs(analytic) + 0.001
        )
        updated = [x.detach().clone() for x in variables]
        updated[index] -= 1e-4 * direction
        assert loss(updated, False).item() < value.item()


def test_energy_and_both_losses_share_backward(module):
    tensors = inputs(sample())
    tensors["atom_types"] = tensors["atom_types"].requires_grad_()
    tensors["positions"] = tensors["positions"].requires_grad_()
    tensors["lattice"] = tensors["lattice"].requires_grad_()
    output = module.compute_properties(
        **tensors, include_forces=True, include_stresses=True, create_graph=True
    )
    loss = (
        output["total_energy"].mean()
        + output["forces"].square().mean()
        + output["stresses"].diagonal(dim1=-2, dim2=-1).mean().square()
    )
    loss.backward()
    for key in ("atom_types", "positions", "lattice"):
        assert torch.isfinite(tensors[key].grad).all()


def test_energy_coordinate_and_strain_derivatives(module):
    tensors = inputs(sample())
    out = module.compute_properties(
        **tensors, include_forces=True, include_stresses=True
    )
    direction = torch.tensor([[0.2, -0.3, 0.4], [-0.2, 0.3, -0.4]])
    expected_force = -(out["forces"] * direction).sum().item()
    strain = torch.diag(torch.tensor([0.2, -0.1, 0.3]))
    expected_strain = (
        (out["stresses"][0] * strain).sum().item()
        * torch.linalg.det(tensors["lattice"])[0].item()
        * GPa
    )
    for kind, expected in (("coordinate", expected_force), ("strain", expected_strain)):
        estimates = []
        for step in (3e-3, 1e-3, 3e-4):
            energies = []
            for sign in (1, -1):
                perturbed = {k: v.clone() for k, v in tensors.items()}
                if kind == "coordinate":
                    perturbed["positions"] += sign * step * direction
                else:
                    transform = torch.eye(3) + sign * step * strain
                    perturbed["positions"] = perturbed["positions"] @ transform
                    perturbed["lattice"] = perturbed["lattice"] @ transform
                energies.append(module.compute_energy(**perturbed).item())
            estimates.append((energies[0] - energies[1]) / (2 * step))
        print("energy derivative", kind, expected, estimates)
        assert (
            min(abs(value - expected) for value in estimates)
            < 0.005 * abs(expected) + 0.001
        )


def test_energy_backward_reaches_composition_coordinates_and_lattice(module):
    tensors = inputs(sample())
    for key in ("atom_types", "positions", "lattice"):
        tensors[key].requires_grad_()
    energy = module.compute_energy(**tensors)
    energy.sum().backward()
    for key in ("atom_types", "positions", "lattice"):
        gradient = tensors[key].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
