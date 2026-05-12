import json
import os
import tempfile
import zipfile
from typing import Dict, List, Tuple, Union

import dgl
import numpy as np
import requests
import torch
import torch.nn as nn
from jarvis.core.atoms import Atoms
from tqdm import tqdm

from src.models.alignn4inv import ALIGNN4inverse
from src.utils.common import get_master_dir
from third_party.alignn_minimal.models.alignn import ALIGNNConfig


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
        ],
        "jv_bulk_modulus_kv_alignn": [
            "https://zenodo.org/records/15772850/files/jv_bulk_modulus_kv_alignn.zip?download=1",
            1,
        ],
        "jv_shear_modulus_gv_alignn": [
            "https://zenodo.org/records/15772850/files/jv_shear_modulus_gv_alignn.zip?download=1",
            1,
        ],
        "jv_magmom_alignn": [
            "https://figshare.com/ndownloader/files/31458685",
            1,
        ],
    }


def _candidate_zip_paths(model_name: str, weights_dir: Union[str, None]) -> List[str]:
    zfile = model_name + ".zip"
    candidates = []
    if weights_dir:
        candidates.append(os.path.join(weights_dir, zfile))
    candidates.append(
        os.path.join(get_master_dir(), "models", "alignn_pretrained", zfile)
    )
    candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "alignn", zfile))
    candidates.append(os.path.join(os.getcwd(), zfile))
    return candidates


def _download_alignn_zip(model_name: str, url: str, dest_path: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(dest_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()


def get_figshare_model_for_inverse_problem(
    model_name, device: Union[str, torch.device, None] = None, weights_dir: Union[str, None] = None
):
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
    if weights_dir:
        weights_dir = os.path.expanduser(weights_dir)
    model_list = figshare_model_list()
    if model_name not in model_list:
        raise ValueError(f"Unknown ALIGNN model name: {model_name}")

    url = model_list[model_name][0]
    zfile = model_name + ".zip"
    path = None
    for candidate in _candidate_zip_paths(model_name, weights_dir):
        if os.path.isfile(candidate):
            if zipfile.is_zipfile(candidate):
                path = candidate
                break
            try:
                os.remove(candidate)
            except OSError:
                pass
    if path is None:
        if weights_dir:
            download_dir = weights_dir
        else:
            download_dir = os.path.join(get_master_dir(), "models", "alignn_pretrained")
        os.makedirs(download_dir, exist_ok=True)
        path = os.path.join(download_dir, zfile)
        try:
            _download_alignn_zip(model_name, url, path)
        except Exception as exc:
            candidate_paths = _candidate_zip_paths(model_name, weights_dir)
            raise RuntimeError(
                "ALIGNN pretrained zip not found or failed to download. "
                f"Model: {model_name}. Expected one of: {candidate_paths}. "
                "You can also pass --alignn_weights_dir to point to a directory "
                f"containing {zfile}."
            ) from exc
        if not zipfile.is_zipfile(path):
            raise RuntimeError(
                f"Downloaded file is not a valid zip: {path}. "
                "Please re-download or provide a correct zip via --alignn_weights_dir."
            )

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


def Load_Pretrained_ALIGNN(
    target_properties,
    device: Union[str, torch.device, None] = None,
    weights_dir: Union[str, None] = None,
):
    if target_properties == "bg_and_eform":
        print("Loading bandgap and formation energy model trained on MEGNet dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_gappbe_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_form_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
    elif target_properties.startswith("bg_and_eform"):
        print(f"Loading bandgap and formation energy model trained on MEGNet dataset for test: {target_properties}")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_gappbe_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_form_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
    elif target_properties == "tc_and_efrom":
        print("Loading Tc predictor trained on JARVIS-SC and formation energy model trained on MEGNet dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_tc_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_form_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
    elif target_properties == "bg_and_ehull":
        print("Loading bandgap predictor trained on MEGNet dataset and E_hull predictor trained on JARVIS DFT dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_gappbe_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_hull_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
    elif target_properties == "B_and_eform":
        print("Loading Bulk Modulus predictor trained on JARVIS DFT dataset and formation energy model trained on MEGNet dataset")
        bandgap_model = get_figshare_model_for_inverse_problem(
            model_name="mp_B_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
        e_form_model = get_figshare_model_for_inverse_problem(
            model_name="mp_e_form_alignnn",
            device=device,
            weights_dir=weights_dir,
        )
    else:
        raise ValueError(
            f"target_properties should be one of the following: bg_and_eform, tc_and_efrom, bg_and_ehull, B_and_eform. current value is {target_properties}"
        )

    return bandgap_model, e_form_model
