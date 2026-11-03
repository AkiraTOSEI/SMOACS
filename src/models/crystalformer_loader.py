import os

import torch

from src.crystalformer.models.latticeformer import Latticeformer
from src.crystalformer.utils import Params
from src.utils.common import get_master_dir


def load_pretrained_crystalformer(
    task: str, layer: int, weight_path_dir: str, device: str = "cuda"
) -> torch.nn.Module:
    weight_path = os.path.join(weight_path_dir, f"megnet-{task}-layer{layer}.ckpt")
    config_path = os.path.join(weight_path_dir, "default.json")
    params = Params(config_path)
    if torch.device(device).type == "cpu":
        params.use_cuda_code = False
    model = Latticeformer(params)
    ##
    ## モデルの重み読み込み
    ##

    model = Latticeformer(params)

    with open(weight_path, "rb") as f:
        ckeckpoint = torch.load(f, map_location="cpu")
        state_dict = ckeckpoint["state_dict"]
        target_std = ckeckpoint["state_dict"]["target_std"]
        target_mean = ckeckpoint["state_dict"]["target_mean"]
        model_name = "model."
        model_name = "swa_model.module."

        model_dict = {
            key.replace(model_name, ""): state_dict[key]
            for key in state_dict
            if key.startswith(model_name)
        }
        model.load_state_dict(model_dict)
        # correct the last linear layer weights
        model.mlp[-1].load_state_dict(
            {
                "weight": model.mlp[-1].weight * target_std[:, None],
                "bias": model.mlp[-1].bias * target_std + target_mean,
            }
        )

    return model


def Load_Pretrained_Crystalformers(device: str = "cuda"):
    # モデルの読み込み
    bandgap_model = load_pretrained_crystalformer(
        task="bandgap",  # 'e_form' or 'bandgap'
        layer=4,
        weight_path_dir=os.path.join(get_master_dir(), f"models/crystalformer"),
        device=device,
    ).to(device)
    e_form_model = load_pretrained_crystalformer(
        task="e_form",  # 'e_form' or 'bandgap'
        layer=4,
        weight_path_dir=os.path.join(get_master_dir(), f"models/crystalformer"),
        device=device,
    ).to(device)
    return bandgap_model, e_form_model
