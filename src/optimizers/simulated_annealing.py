from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import hashlib
import torch


def _clone_state(state: List[torch.Tensor]) -> List[torch.Tensor]:
    return [t.clone() for t in state]


def _state_hash(state: List[torch.Tensor], decimals: int = 4) -> str:
    parts = []
    for tensor in state:
        rounded = torch.round(tensor.detach().cpu(), decimals=decimals)
        parts.append(rounded.numpy().tobytes())
    return hashlib.md5(b"".join(parts)).hexdigest()


def _sample_loss(eval_result: Dict[str, Any]) -> torch.Tensor:
    if "each_loss" not in eval_result:
        raise ValueError("per-sample SA acceptance requires eval_result['each_loss']")
    each_loss = eval_result["each_loss"]
    if not torch.is_tensor(each_loss):
        raise ValueError("eval_result['each_loss'] must be a torch.Tensor")
    if each_loss.ndim == 0:
        return each_loss.view(1)
    if each_loss.ndim != 1:
        raise ValueError(
            "eval_result['each_loss'] must be 1D for per-sample SA acceptance, "
            f"got shape={tuple(each_loss.shape)}"
        )
    return each_loss


def _row_accept_mask(
    accept_mask: torch.Tensor,
    sample_sizes: Optional[torch.Tensor],
    row_count: int,
) -> torch.Tensor:
    if sample_sizes is None:
        if row_count != accept_mask.numel():
            raise ValueError(
                "sample_sizes_by_state is required for state tensors whose first "
                f"dimension is not the sample count: row_count={row_count}, "
                f"sample_count={accept_mask.numel()}"
            )
        return accept_mask

    sizes = sample_sizes.to(device=accept_mask.device, dtype=torch.long)
    if sizes.ndim != 1:
        raise ValueError(f"sample sizes must be 1D, got shape={tuple(sizes.shape)}")
    if sizes.numel() != accept_mask.numel():
        raise ValueError(
            "sample sizes length must match sample count: "
            f"{sizes.numel()} != {accept_mask.numel()}"
        )
    if int(sizes.sum().item()) != row_count:
        raise ValueError(
            "sample sizes must sum to the state row count: "
            f"sum={int(sizes.sum().item())}, row_count={row_count}"
        )
    return torch.repeat_interleave(accept_mask, sizes)


def _merge_state_by_sample(
    current_state: List[torch.Tensor],
    proposal_state: List[torch.Tensor],
    accept_mask: torch.Tensor,
    sample_sizes_by_state: List[Optional[torch.Tensor]],
) -> List[torch.Tensor]:
    if len(sample_sizes_by_state) != len(current_state):
        raise ValueError(
            "sample_sizes_by_state length must match state length: "
            f"{len(sample_sizes_by_state)} != {len(current_state)}"
        )

    merged_state: List[torch.Tensor] = []
    for current, proposal, sample_sizes in zip(
        current_state, proposal_state, sample_sizes_by_state
    ):
        if current.shape != proposal.shape:
            raise ValueError(
                "current and proposal states must have matching shapes: "
                f"{tuple(current.shape)} != {tuple(proposal.shape)}"
            )
        if current.ndim == 0:
            raise ValueError("per-sample SA does not support scalar state tensors")

        row_mask = _row_accept_mask(
            accept_mask=accept_mask,
            sample_sizes=sample_sizes,
            row_count=current.shape[0],
        ).to(device=current.device)
        view_shape = (current.shape[0],) + (1,) * (current.ndim - 1)
        merged_state.append(torch.where(row_mask.view(view_shape), proposal, current))
    return merged_state


def geometric_schedule(t0: float, tmin: float, alpha: float) -> Callable[[int], float]:
    def _schedule(step: int) -> float:
        return max(tmin, t0 * (alpha**step))

    return _schedule


def geometric_schedule_with_endpoints(
    t0: float, tmin: float, num_steps: int
) -> Callable[[int], float]:
    if not isinstance(num_steps, int):
        raise ValueError(f"num_steps must be int, got {type(num_steps).__name__}")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    validated: Dict[str, float] = {}
    for name, value in (("t0", t0), ("tmin", tmin)):
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number, got {value!r}") from None
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if scalar <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value!r}")
        validated[name] = scalar
    t0_scalar = validated["t0"]
    tmin_scalar = validated["tmin"]
    if tmin_scalar > t0_scalar:
        raise ValueError(f"tmin must be <= t0, got tmin={tmin!r}, t0={t0!r}")
    if num_steps == 1:
        return lambda _step: t0_scalar

    ratio = tmin_scalar / t0_scalar

    def _schedule(step: int) -> float:
        if step <= 0:
            return t0_scalar
        if step >= num_steps - 1:
            return tmin_scalar
        exponent = step / float(num_steps - 1)
        return t0_scalar * (ratio**exponent)

    return _schedule


@dataclass
class SAHistoryEntry:
    step: int
    temperature: float
    loss: float
    best_loss: float
    accept_rate: float


class SimulatedAnnealingOptimizer:
    def __init__(
        self,
        evaluate_fn: Callable[[List[torch.Tensor], int], Dict[str, Any]],
        proposal_fn: Callable[
            [List[torch.Tensor], torch.Generator], List[torch.Tensor]
        ],
        schedule: Callable[[int], float],
        seed: Optional[int] = None,
        cache_size: int = 0,
        hash_decimals: int = 4,
    ) -> None:
        self.evaluate_fn = evaluate_fn
        self.proposal_fn = proposal_fn
        self.schedule = schedule
        self.seed = seed
        self.cache_size = cache_size
        self.hash_decimals = hash_decimals
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.best_eval: Optional[Dict[str, Any]] = None

        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def _eval(self, state: List[torch.Tensor], step_i: int) -> Dict[str, Any]:
        if self.cache_size <= 0:
            return self.evaluate_fn(state, step_i)
        key = _state_hash(state, decimals=self.hash_decimals)
        if key in self._cache:
            return self._cache[key]
        result = self.evaluate_fn(state, step_i)
        if len(self._cache) >= self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result

    def run(
        self,
        initial_state: List[torch.Tensor],
        steps: int,
        trials_per_step: int = 1,
        per_sample_accept: bool = False,
        sample_sizes_by_state: Optional[List[Optional[torch.Tensor]]] = None,
        step_callback: Optional[
            Callable[
                [
                    int,
                    List[torch.Tensor],
                    Dict[str, Any],
                    Dict[str, Any],
                    Dict[str, Any],
                ],
                None,
            ]
        ] = None,
    ) -> Tuple[List[torch.Tensor], List[SAHistoryEntry]]:
        current_state = _clone_state(initial_state)
        current_eval = self._eval(current_state, 0)
        best_state = _clone_state(current_state)
        best_eval = current_eval
        self.best_eval = best_eval
        history: List[SAHistoryEntry] = []
        best_sample_loss: Optional[torch.Tensor] = None
        if per_sample_accept:
            current_sample_loss = _sample_loss(current_eval)
            best_sample_loss = current_sample_loss.detach().clone()
            if sample_sizes_by_state is None:
                sample_sizes_by_state = [None] * len(initial_state)

        for step_i in range(steps):
            temperature = self.schedule(step_i)
            accepted = 0.0
            accept_flags: List[int] = []
            sample_accept_flags: List[List[int]] = []
            for _ in range(trials_per_step):
                proposal = self.proposal_fn(current_state, self._generator)
                proposal_eval = self._eval(proposal, step_i)

                if per_sample_accept:
                    current_each_loss = _sample_loss(current_eval)
                    proposal_each_loss = _sample_loss(proposal_eval)
                    if current_each_loss.shape != proposal_each_loss.shape:
                        raise ValueError(
                            "current and proposal each_loss shapes must match: "
                            f"{tuple(current_each_loss.shape)} != "
                            f"{tuple(proposal_each_loss.shape)}"
                        )
                    delta = proposal_each_loss - current_each_loss
                    rand = torch.rand(delta.shape, generator=self._generator).to(
                        delta.device
                    )
                    accept_mask = (delta <= 0) | (
                        torch.exp(-delta / temperature) > rand
                    )
                    accepted += float(accept_mask.sum().item())
                    accept_flags.append(int(accept_mask.any().item()))
                    sample_accept_flags.append(
                        [int(flag) for flag in accept_mask.detach().cpu().tolist()]
                    )

                    if bool(accept_mask.all().item()):
                        current_state = proposal
                        current_eval = proposal_eval
                    elif bool(accept_mask.any().item()):
                        assert sample_sizes_by_state is not None
                        current_state = _merge_state_by_sample(
                            current_state=current_state,
                            proposal_state=proposal,
                            accept_mask=accept_mask,
                            sample_sizes_by_state=sample_sizes_by_state,
                        )
                        current_eval = self._eval(current_state, step_i)

                    assert best_sample_loss is not None
                    current_each_loss = _sample_loss(current_eval).detach()
                    improved_mask = current_each_loss < best_sample_loss.to(
                        current_each_loss.device
                    )
                    if bool(improved_mask.any().item()):
                        assert sample_sizes_by_state is not None
                        best_state = _merge_state_by_sample(
                            current_state=best_state,
                            proposal_state=current_state,
                            accept_mask=improved_mask,
                            sample_sizes_by_state=sample_sizes_by_state,
                        )
                        best_sample_loss = torch.where(
                            improved_mask,
                            current_each_loss,
                            best_sample_loss.to(current_each_loss.device),
                        )
                        best_eval = self._eval(best_state, step_i)
                        self.best_eval = best_eval
                    continue

                delta = proposal_eval["total_loss"] - current_eval["total_loss"]
                if delta <= 0:
                    accept = True
                else:
                    rand = torch.rand((), generator=self._generator).item()
                    accept = (torch.exp(-delta / temperature)).item() > rand
                if accept:
                    current_state = proposal
                    current_eval = proposal_eval
                    accepted += 1.0
                    accept_flags.append(1)
                    if current_eval["total_loss"] < best_eval["total_loss"]:
                        best_state = _clone_state(current_state)
                        best_eval = current_eval
                        self.best_eval = best_eval
                else:
                    accept_flags.append(0)
            if per_sample_accept:
                sample_count = max(_sample_loss(current_eval).numel(), 1)
                accept_rate = accepted / max(trials_per_step * sample_count, 1)
                assert best_sample_loss is not None
                best_loss = float(best_sample_loss.detach().mean().cpu())
            else:
                accept_rate = accepted / max(trials_per_step, 1)
                best_loss = float(best_eval["total_loss"].detach().cpu())
            entry = SAHistoryEntry(
                step=step_i,
                temperature=temperature,
                loss=float(current_eval["total_loss"].detach().cpu()),
                best_loss=best_loss,
                accept_rate=accept_rate,
            )
            history.append(entry)
            if step_callback is not None:
                step_callback(
                    step_i,
                    current_state,
                    current_eval,
                    best_eval,
                    {
                        "temperature": temperature,
                        "accept_rate": accept_rate,
                        "accept_flags": accept_flags,
                        "sample_accept_flags": sample_accept_flags,
                    },
                )

        return best_state, history
