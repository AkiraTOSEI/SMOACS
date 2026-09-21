from typing import Any, Dict, List

import torch

from src.models.alignn_loader import get_figshare_model_for_inverse_problem
from src.models.crystalformer_loader import load_pretrained_crystalformer
from src.predictors.alignn_predictor import AlignnPredictor
from src.predictors.crystalformer_predictor import CrystalformerPredictor
from src.utils.common import get_master_dir


def build_predictors(predictor_configs: List[Dict[str, Any]], device: str) -> Dict[str, Any]:
    predictors: Dict[str, Any] = {}
    for cfg in predictor_configs:
        name = cfg["name"]
        if name in predictors:
            raise ValueError(f"Duplicate predictor name: {name}")
        p_type = cfg["type"].lower()
        params = cfg.get("params", {})
        clip_min = params.get("clip_min")

        if p_type == "alignn":
            checkpoint = cfg["checkpoint"]
            weights_dir = params.get("weights_dir")
            model = get_figshare_model_for_inverse_problem(
                checkpoint, weights_dir=weights_dir
            )
            model = model.to(device)
            predictors[name] = AlignnPredictor(name=name, model=model, clip_min=clip_min)
        elif p_type == "crystalformer":
            task = params.get("task")
            layer = params.get("layer", 4)
            if task is None:
                raise ValueError("crystalformer predictor requires params.task")
            weight_path_dir = params.get(
                "weight_path_dir",
                f"{get_master_dir()}/models/crystalformer",
            )
            model = load_pretrained_crystalformer(
                task=task,
                layer=layer,
                weight_path_dir=weight_path_dir,
                device=device,
            )
            model = model.to(device)
            predictors[name] = CrystalformerPredictor(
                name=name, model=model, clip_min=clip_min
            )
        else:
            raise ValueError(f"Unsupported predictor type: {p_type}")

    return predictors
