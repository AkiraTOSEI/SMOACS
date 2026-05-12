from typing import Dict, Optional

import torch


class CrystalformerPredictor:
    def __init__(self, name: str, model: torch.nn.Module, clip_min: Optional[float] = None) -> None:
        self.name = name
        self.model = model
        self.clip_min = clip_min
        if self.model.training:
            self.model.eval()

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        normalized_dist = inputs["normalized_dist"]
        pos = inputs["pos"]
        batch = inputs["batch"]
        trans_vec = inputs["trans_vec"]
        size = inputs["size"]
        pred = self.model((normalized_dist, pos, batch, trans_vec, size))
        if self.clip_min is not None:
            pred = torch.clamp(pred, min=self.clip_min)
        return pred
