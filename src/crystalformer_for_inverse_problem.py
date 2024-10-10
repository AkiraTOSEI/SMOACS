import os

import torch

from .crystalformer.models.latticeformer import Latticeformer
from .crystalformer.utils import Params


def load_pretrained_crystalformer(
    task: str,
    layer: int,
    device: str,
    weight_path_dir: str
):

    weight_path = os.path.join(weight_path_dir, f"megnet-{task}-layer{layer}.ckpt")
    config_path = os.path.join(weight_path_dir, "default.json")
    params = Params(config_path)
    model = Latticeformer(params)
    ##
    ## モデルの重み読み込み
    ## 


    model = Latticeformer(params)
    

    with open(weight_path , "rb") as f:
        ckeckpoint = torch.load(f, map_location='cpu')
        state_dict = ckeckpoint['state_dict']
        target_std = ckeckpoint['state_dict']['target_std']
        target_mean = ckeckpoint['state_dict']['target_mean']
        model_name = "model."
        model_name = "swa_model.module."

        model_dict = { key.replace(model_name, ""):state_dict[key] for key in state_dict if key.startswith(model_name) }
        model.load_state_dict(model_dict)
        # correct the last linear layer weights
        model.mlp[-1].load_state_dict({
            'weight': model.mlp[-1].weight * target_std[:,None],
            'bias': model.mlp[-1].bias * target_std + target_mean,
        })

    return model

def Load_Pretrained_Crystalformers(
        prediction_loss_setting_dict:dict,
) -> dict:
    """
    Loads pretrained Crystalformer models for predicting bandgap and formation energy.
    Args:
        device (str): The device to load the models onto, e.g., 'cpu' or 'cuda'.
        task (str): The task to load the models for, either 'bandgap' or 'e_form'.
    Returns:
        dict: A dictionary containing the pretrained Crystalformer models.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    for loss_key in prediction_loss_setting_dict.keys():
        prediction_loss_setting_dict[loss_key]['prediction_model'] = load_pretrained_crystalformer(
            task = loss_key, # 'e_form' or 'bandgap'
            layer = 4,
            device=device,
            weight_path_dir = 'models/crystalformer',
        ).to(device)

    return prediction_loss_setting_dict
