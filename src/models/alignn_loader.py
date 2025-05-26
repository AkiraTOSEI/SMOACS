# from alignn.data import get_torch_dataset
import json
import os
import tempfile
import zipfile
from typing import Tuple, Union

import dgl
import numpy as np
import requests
import torch
import torch.nn as nn
from jarvis.core.atoms import Atoms
from tqdm import tqdm

from alignn.models.alignn import ALIGNNConfig
from src.models.alignn4inv import ALIGNN4inverse


def figshare_model_list():
    return {
        "mp_e_form_alignnn": [
            "https://figshare.com/ndownloader/files/31458811", # E_form predictor trained on MEGNet
            1,
        ],
        "mp_gappbe_alignnn": [
            "https://figshare.com/ndownloader/files/31458814", # bandgap predictor trained on MEGNet
            1,
        ],
         "mp_tc_alignnn": [
            "https://figshare.com/ndownloader/files/38789199", # Tc predictor trained on JARVIS supercon
            1,
        ],
        "mp_e_hull_alignnn": [
            "https://figshare.com/ndownloader/files/31458658", # E_hull predictor trained on JARVIS DFT
            1,
        ],
        "mp_B_alignnn": [
            "https://figshare.com/ndownloader/files/31458649", # Bulk Modulus predictor trained on JARVIS DFT
            1,
        ]

    }


def get_figshare_model_for_inverse_problem(model_name):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tmp = figshare_model_list()[model_name]
    url = tmp[0]
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname("__file__"), zfile))
    # path = str(os.path.join(os.path.dirname(__file__), zfile)) # 非対話モードであれば、こちらを使う
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    cfg = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            chks.append(i)
        if "config.json" in i:
            cfg = i
        if "best_model.pt" in i:
            tmp = i
            chks.append(i)

    print("Using chk file", tmp, "from ", chks)
    print("Path", os.path.abspath(path))
    print("Config", os.path.abspath(cfg))
    config = json.loads(zipfile.ZipFile(path).read(cfg))
    # print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
    data = zipfile.ZipFile(path).read(tmp)
    model = ALIGNN4inverse(ALIGNNConfig(**config["model"]))

    # new_file, filename = tempfile.mkstemp()
    filename = "tmp.pt"
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)

    return model


def Load_Pretrained_ALIGNN(target_properties):
    if target_properties == "bg_and_eform":
        print("Loading bandgap and formation energy model trained on MEGNet dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_gappbe_alignnn"
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_form_alignnn"
        )
    elif target_properties == "tc_and_efrom":
        print("Loading Tc predictor trained on JARVIS-SC and formation energy model trained on MEGNet dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_tc_alignnn"
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_form_alignnn"
        )
    elif target_properties == "bg_and_ehull":
        print("Loading bandgap predictor trained on MEGNet dataset and E_hull predictor trained on JARVIS DFT dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_gappbe_alignnn"
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_hull_alignnn"
        )
    elif target_properties == "B_and_eform":
        print("Loading Bulk Modulus predictor trained on JARVIS DFT dataset and formation energy model trained on MEGNet dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_B_alignnn"
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_form_alignnn"
        )
    else:
        raise ValueError(
            f"target_properties should be one of the following: bg_and_eform, tc_and_efrom, bg_and_ehull, B_and_eform. current value is {target_properties}"
        )

    return bandgap_model, e_form_model
