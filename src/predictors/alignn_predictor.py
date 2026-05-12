from typing import Dict, Optional

import torch


class AlignnPredictor:
    def __init__(self, name: str, model: torch.nn.Module, clip_min: Optional[float] = None) -> None:
        self.name = name
        self.model = model
        self.clip_min = clip_min
        if self.model.training:
            self.model.eval()

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        g = inputs["g"]
        lg = inputs["lg"]
        atomic_features = inputs["atomic_features"]
        bondlength = inputs["bondlength"]
        bond_angles = inputs["bond_angles"]
        pred = self.model((g, lg), atomic_features, bondlength, bond_angles)
        if self.clip_min is not None:
            pred = torch.clamp(pred, min=self.clip_min)
        return pred
