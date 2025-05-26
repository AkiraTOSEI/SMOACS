## losses/gap_loss.py
from typing import Optional

import torch


class GapLoss(torch.nn.Module):
    """
    A custom loss function for controlling band gap predictions.

    Supports three modes:
        - 'band': penalize values outside [gap_min, gap_max]
        - 'minimization': minimization. if gap_min is not None, penalize only when loss value > gap_min
        - 'maximization': maximization. if gap_max is not None, penalize only when loss value < gap_max

    Args:
        gap_min (Optional[float]): Minimum target value. Required for 'min' or 'band' mode.
        gap_max (Optional[float]): Maximum target value. Required for 'max' or 'band' mode.
        mode (str): One of ['band', 'min', 'max']. Determines loss behavior.

    Raises:
        ValueError: If mode or required values are inconsistent.
    """

    def __init__(
        self,
        gap_min: Optional[float] = None,
        gap_max: Optional[float] = None,
        mode: str = "band",
    ):
        super().__init__()
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.mode = mode

        if mode not in ["band", "minimization", "maximization"]:
            raise ValueError("mode must be 'band', 'minimization' or 'maximization'")

        if mode == "band" and gap_min is not None and gap_max is not None:
            self.mean = (gap_min + gap_max) / 2
            self.margin = (gap_max - gap_min) / 2
        elif mode == "minimization" or mode == "maximization":
            pass
        else:
            raise ValueError(
                f"mode and gap_min, gap_max are not matched! mode: {mode}, gap_min: {gap_min}, gap_max: {gap_max}"
            )

    def forward(self, x: torch.Tensor, reduction="mean") -> torch.Tensor:
        if self.mode == "band":
            loss = self.band_loss(x)
        elif self.mode == "minimization":
            loss = self.min_loss(x)
        elif self.mode == "maximization":
            loss = self.max_loss(x)
        else:
            raise ValueError("mode must be 'band', 'minimization' or 'maximization'")

        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "sum":
            return torch.sum(loss)
        elif reduction == "none":
            return loss
        else:
            raise ValueError("reduction must be 'mean', 'sum' or 'none")

    def min_loss(self, x):
        # (english) minimization x
        if self.gap_min is None:
            return x
        else:
            # (english) minimization x to gap_min. If x < gap_min, loss is 0.
            return torch.clip(x - self.gap_min, min=0).squeeze()

    def max_loss(self, x):
        # (english) maximization x
        if self.gap_max is None:
            return -x
        else:
            # (english) maximization x to gap_max. If x > gap_max, loss is 0.
            return torch.clip(self.gap_max - x, min=0).squeeze()

    def band_loss(self, x):
        return torch.clip(torch.abs(x - self.mean) - self.margin, min=0).squeeze()
