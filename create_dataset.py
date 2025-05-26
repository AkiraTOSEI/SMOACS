import os
import random
from typing import Optional

import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
from tqdm import tqdm

from src.utils.structure_neutrality_check  import count_elements, elec_neutral_check_SUPER_COMMON


def create_initial_structure_from_database(
    base_dir_path: str,
    num_candidates: int,
    database_name: str,
    max_atoms: Optional[int] = None,
):
    """
    Processes data from the 'megnet' or 'dft_3d' dataset, saving electrically neutral and non-neutral structures separately.

    Args:
        base_dir_path (str): The base directory path where the files will be stored.
        num_candidates (int): The number of candidate structures to save.

    """
    if database_name == "megnet":
        id_tag = "id"
    elif database_name == "dft_3d":
        id_tag = "jid"
    else:
        raise ValueError(
            f"Database name must be either 'megnet' or 'dft_3d'. Got: {database_name}"
        )

    dataset = jarvis_data(database_name)
    random.shuffle(dataset)

    # Save without considering electrical neutrality
    if max_atoms is None:
        max_atoms = 100000000
        max_strings = ""
    else:
        max_strings = f"_max_{max_atoms}atoms"

    # Save considering electrical neutrality by common
    saved_dir = os.path.join(
        base_dir_path, f"initial_candidates_from_{database_name}{max_strings}_neutral"
    )
    os.makedirs(saved_dir, exist_ok=True)
    # clean the directory
    for file in os.listdir(saved_dir):
        os.remove(os.path.join(saved_dir, file))

    num_neutral = 0
    for data in dataset:
        atoms, stoichs = [], []
        for _atom, _stoich in count_elements(data["atoms"]["elements"]).items():
            atoms.append(_atom)
            stoichs.append([_stoich])

        # 最大の原子数を超える場合はスキップ
        if np.sum(stoichs) > max_atoms:
            continue

        is_neutral, _, _ = elec_neutral_check_SUPER_COMMON(
            num_neutral, num_candidates, elements=atoms, stoichs=stoichs
        )
        if is_neutral:
            Atoms.from_dict(data["atoms"]).write_poscar(
                os.path.join(saved_dir, f"{data[id_tag]}.vasp")
            )
            num_neutral += 1

        if num_neutral >= num_candidates:
            break

    return


def create_random_perovskite(
    num_candidate: int,
    abc_max: float,
    abc_min: float,
    mode: str,
    base_dir_path: str,
    repeat4supercells: Optional[int] = None,
):
    if repeat4supercells is not None:
        save_dir = os.path.join(
            base_dir_path,
            f"initial_{mode}_perovskite_{repeat4supercells}x{repeat4supercells}x{repeat4supercells}",
        )
    else:
        save_dir = os.path.join(base_dir_path, f"initial_{mode}_perovskite")

    # create and clean the directory
    os.makedirs(save_dir, exist_ok=True)
    for file in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, file))

    perov_coords = np.array(
        [
            [0.5, 0.5, 0.5],  # B site (Ti) +4
            [0.0, 0.0, 0.0],  # A site (Ba) +2
            [0.5, 0.0, 0.0],  # X site (O) -2
            [0.0, 0.5, 0.0],  # X site (O) -2
            [0.0, 0.0, 0.5],  # X site (O) -2
        ]
    )
    if mode == "random":
        abc = np.random.random((num_candidate, 3)) * (abc_max - abc_min) + abc_min
    elif mode.startswith("normal"):
        std = float(mode.replace("normal", ""))
        base_length = (
            np.random.random((num_candidate, 1)) * (abc_max - abc_min) * 0.75 + abc_min
        )
        ratio = (
            np.random.normal(0, std, num_candidate * 3).reshape(num_candidate, 3) + 1.0
        )
        abc = np.clip(base_length * ratio, a_min=abc_min, a_max=abc_max)
    else:
        raise ValueError(f"Mode {mode} is not implemented.")

    for cand_i in tqdm(
        range(num_candidate), desc="Creating random perovskite structures"
    ):
        atom_dic = {
            "lattice_mat": np.diag(abc[cand_i]),
            "coords": perov_coords,
            "elements": ["Ti", "Ba", "O", "O", "O"] * 5,  # dummy atom
            "abc": abc[cand_i].tolist(),
            "angles": [90, 90, 90],
            "cartesian": False,
            "props": [""] * 5,
        }
        atoms_data = Atoms.from_dict(atom_dic)
        if repeat4supercells is not None:
            atoms_data = atoms_data.make_supercell(
                (repeat4supercells, repeat4supercells, repeat4supercells)
            )

        atoms_data.write_poscar(os.path.join(save_dir, f"perovskite_{cand_i}.vasp"))

    return


if __name__ == "__main__":
    base_dir_path = "./data/raw_data"
    num_candidates = 256
    max_atoms = 10
    random.seed(42)
    os.makedirs(base_dir_path, exist_ok=True)

    ### create random perovskite structure
    create_random_perovskite(
        num_candidate=num_candidates,
        mode="random",
        abc_max=10,
        abc_min=2,
        base_dir_path=base_dir_path,
    )

    ### create random 3x3x3 perovskite structure
    create_random_perovskite(
        num_candidate=num_candidates,
        mode="random",
        abc_max=10,
        abc_min=2,
        base_dir_path=base_dir_path,
        repeat4supercells=3,
    )

    ## create initial structure from megnet dataset
    create_initial_structure_from_database(
        base_dir_path, num_candidates, "megnet", max_atoms=max_atoms
    )
