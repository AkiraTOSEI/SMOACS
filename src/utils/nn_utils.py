import torch


def temperature_softmax(logits, temperature=1.0, dim=-1):
    """Applies a temperature-scaled softmax to the input logits.

    This function modifies the softmax operation by scaling the logits with
    a temperature parameter before applying softmax. The temperature parameter
    can adjust the sharpness of the output distribution. A higher temperature
    makes the distribution more uniform, while a lower temperature makes it
    sharper.

    Args:
        logits (torch.Tensor): The input logits to which softmax will be applied.
        temperature (float, optional): The temperature to scale the logits. Default is 1.0.

    Returns:
        torch.Tensor: The softmax output after applying temperature scaling.

    Raises:
        ValueError: If the temperature is non-positive.

    Example:
        >>> logits = torch.tensor([2.0, 1.0, 0.1])
        >>> temperature = 0.5
        >>> softmax_outputs = temperature_softmax(logits, temperature)
        >>> print(softmax_outputs)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    # Adjust logits based on the temperature
    adjusted_logits = logits / temperature
    # Apply softmax to the adjusted logits
    return torch.softmax(adjusted_logits, dim=dim)
