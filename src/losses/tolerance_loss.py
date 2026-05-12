import torch
import torch.nn as nn


class ToleranceLoss(nn.Module):
    """
    A custom loss function that penalizes predictions outside a specified tolerance range.

    The loss is zero when predictions are within [low, high], and increases linearly
    outside this range, with an optional margin softening the boundary.

    Args:
        low (float): Lower bound of the valid prediction range (default: 0.8).
        high (float): Upper bound of the valid prediction range (default: 1.0).
        margin (float): Margin subtracted/added to the range to define a tolerance zone (default: 0.1).
    """

    def __init__(self, low=0.8, high=1.0, margin=0.1):
        super(ToleranceLoss, self).__init__()
        self.low = low + margin
        self.high = high - margin

    def forward(self, predictions):
        """
        Compute custom loss.

        Args:
            predictions (torch.Tensor): Tensor containing the predicted values.

        Returns:
            torch.Tensor: Tensor containing the computed loss.
        """
        # Ensure predictions are within a valid range by applying thresholds
        lower_mask = predictions < self.low
        upper_mask = predictions > self.high

        # Calculate differences from the nearest boundary
        loss = torch.where(lower_mask, self.low - predictions, predictions - self.high)

        # Apply mask to zero out losses within the acceptable range
        loss = torch.where(
            (predictions >= self.low) & (predictions <= self.high),
            torch.zeros_like(predictions),
            loss,
        )

        return loss
