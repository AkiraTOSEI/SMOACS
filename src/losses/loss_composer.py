from typing import Any, Dict, List, Tuple

import torch

from src.utils.safe_expression import SafeExpressionEvaluator


class LossComposer:
    def __init__(self, terms: List[Dict[str, Any]], evaluator: SafeExpressionEvaluator) -> None:
        self.terms = terms
        self.evaluator = evaluator

    def compute(
        self, pred: Dict[str, torch.Tensor], context: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not pred:
            raise ValueError("pred is empty")
        sample_tensor = next(iter(pred.values()))
        if sample_tensor.ndim == 0:
            num_samples = 1
        else:
            num_samples = sample_tensor.shape[0]
        device = sample_tensor.device

        term_values: Dict[str, torch.Tensor] = {}
        each_loss = torch.zeros(num_samples, device=device)

        variables = {"pred": pred}
        variables.update(context)

        for term in self.terms:
            name = term["name"]
            expr = term["expr"]
            weight = float(term.get("weight", 1.0))
            value = self.evaluator.eval(expr, variables)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=device)
            if value.ndim == 0:
                value = value.repeat(num_samples)
            term_values[name] = value
            each_loss = each_loss + (weight * value)

        total_loss = each_loss.mean()
        return total_loss, each_loss, term_values
