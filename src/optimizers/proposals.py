from typing import Dict, List, Optional

import torch


def _clone_state(state: List[torch.Tensor]) -> List[torch.Tensor]:
    return [t.clone() for t in state]


def _clamp_scaled_abc(
    scaled_batch_abc: torch.Tensor, abc_scaler, device: torch.device
) -> torch.Tensor:
    base_length = abc_scaler.base_length.to(device)
    min_scaled = abc_scaler.min_length / base_length
    max_scaled = abc_scaler.max_length / base_length
    return torch.clamp(scaled_batch_abc, min=min_scaled, max=max_scaled)


def _clamp_scaled_angle(
    scaled_batch_angle: torch.Tensor, angle_scaler, device: torch.device
) -> torch.Tensor:
    return torch.clamp(scaled_batch_angle, min=0.0, max=1.0)


def propose_state(
    state: List[torch.Tensor],
    generator: torch.Generator,
    move_probs: Dict[str, float],
    move_scales: Dict[str, float],
    init_coords: Optional[torch.Tensor] = None,
    limit_coords_displacement: Optional[float] = None,
    abc_scaler=None,
    angle_scaler=None,
    hard_constraints: bool = True,
) -> List[torch.Tensor]:
    """
    Propose a new state by applying a single random move.

    Moves:
      - coords: Gaussian noise on fractional coordinates (wrapped to [0, 1))
      - lattice: Gaussian noise on scaled lattice lengths
      - angle: Gaussian noise on scaled lattice angles
      - atom: Gaussian noise on atomic distribution logits

    Note:
      The current SA integration in `main.py` fixes move choices to
      `coords/lattice/angle`, but this helper keeps `atom` support for
      backward compatibility with older callers.
    """
    proposal = _clone_state(state)
    device = proposal[0].device
    move_names = list(move_probs.keys())
    probs = torch.tensor([move_probs[m] for m in move_names], device="cpu")
    probs = probs / probs.sum()
    move_idx = torch.multinomial(probs, 1, generator=generator).item()
    move = move_names[move_idx]

    if move == "coords":
        sigma = move_scales.get("coords", 1e-3)
        noise = (
            torch.randn(
                proposal[2].shape,
                generator=generator,
                device="cpu",
                dtype=proposal[2].dtype,
            ).to(proposal[2].device)
            * sigma
        )
        proposal[2] = proposal[2] + noise
        if (
            hard_constraints
            and init_coords is not None
            and limit_coords_displacement is not None
        ):
            init_coords = init_coords.to(device)
            proposal[2] = torch.clamp(
                proposal[2],
                min=init_coords - limit_coords_displacement,
                max=init_coords + limit_coords_displacement,
            )
        proposal[2] = torch.remainder(proposal[2], 1.0)
    elif move == "lattice":
        sigma = move_scales.get("lattice", 1e-3)
        noise = (
            torch.randn(
                proposal[0].shape,
                generator=generator,
                device="cpu",
                dtype=proposal[0].dtype,
            ).to(proposal[0].device)
            * sigma
        )
        proposal[0] = proposal[0] + noise
        if hard_constraints and abc_scaler is not None:
            proposal[0] = _clamp_scaled_abc(proposal[0], abc_scaler, device)
    elif move == "angle":
        sigma = move_scales.get("angle", 1e-3)
        noise = (
            torch.randn(
                proposal[1].shape,
                generator=generator,
                device="cpu",
                dtype=proposal[1].dtype,
            ).to(proposal[1].device)
            * sigma
        )
        proposal[1] = proposal[1] + noise
        if hard_constraints and angle_scaler is not None:
            proposal[1] = _clamp_scaled_angle(proposal[1], angle_scaler, device)
    elif move == "atom":
        sigma = move_scales.get("atom", 1e-2)
        noise = (
            torch.randn(
                proposal[3].shape,
                generator=generator,
                device="cpu",
                dtype=proposal[3].dtype,
            ).to(proposal[3].device)
            * sigma
        )
        proposal[3] = proposal[3] + noise
    else:
        raise ValueError(f"Unknown move: {move}")

    return proposal
