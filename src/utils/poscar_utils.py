import os
import shutil

import numpy as np
from jarvis.core.atoms import Atoms

from src.utils.common import create_element_list
from src.utils.coord import compute_lattice_vectors
from src.utils.feature_utils import calculate_onehot
from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)
from src.utils.nn_utils import temperature_softmax


def save_poscar(
    optimized_mini_batch_inputs_dict, tmp_poscar_dir, onehot_temperature: float = 1e-8
):
    """
    最適化途中の構造を保存する関数
    """
    ### crean up tmp_poscar_dir
    if os.path.exists(tmp_poscar_dir):
        shutil.rmtree(tmp_poscar_dir)
    os.makedirs(tmp_poscar_dir, exist_ok=True)

    ### load data from optimized_mini_batch_inputs_dict
    batch_abc = optimized_mini_batch_inputs_dict["batch_abc"].detach().cpu()
    batch_angle = optimized_mini_batch_inputs_dict["batch_angle"].detach().cpu()
    atomic_distribution = (
        optimized_mini_batch_inputs_dict["atomic_distribution"].detach().cpu()
    )
    batch_dir_coords = (
        optimized_mini_batch_inputs_dict["batch_dir_coords"].detach().cpu().numpy()
    )
    normed_batch_dir_coords = np.remainder(
        batch_dir_coords, 1.0
    )  # 内部座標の値を周期的境界条件で0~1に制限する
    size = optimized_mini_batch_inputs_dict["size"].detach().cpu().numpy()
    fnames = optimized_mini_batch_inputs_dict["fnames"]
    atomic_mask = optimized_mini_batch_inputs_dict["atomic_mask"].detach().cpu()
    ox_states_used_mask = (
        optimized_mini_batch_inputs_dict["ox_states_used_mask"].detach().cpu()
    )
    atom_feat_matrix = (
        optimized_mini_batch_inputs_dict["atom_feat_matrix"].detach().cpu()
    )
    ox_mask_learnable_tensor_per_crystal = (
        optimized_mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"]
        .detach()
        .cpu()
    )

    # compute lattice vectors
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle).numpy()
    batch_abc = batch_abc.numpy()
    batch_angle = batch_angle.numpy()

    # Get elements
    ### masking
    if ox_mask_learnable_tensor_per_crystal is not None:
        stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
            ox_mask_learnable_tensor_per_crystal, size
        )
        normalized_dist, sharpened_ox_mask = masking_atomic_distribution(
            atomic_distribution,
            atomic_mask,
            ox_states_used_mask,
            stacked_learnable_ox_weight,
            onehot_temperature,
        )
    else:
        normalized_dist = temperature_softmax(
            atomic_distribution, temperature=onehot_temperature
        )

    ### get onehot atomic distribution
    _, onehot_x, _, _ = calculate_onehot(
        {
            "normalized_dist": normalized_dist,
            "sharpened_ox_mask": sharpened_ox_mask,
        },
        atom_feat_matrix,
    )

    ### create element symbol list
    VALID_ELEMENTS98 = create_element_list()
    element_ids = np.argmax(onehot_x.numpy(), axis=1)
    elements = [VALID_ELEMENTS98[e_id] for e_id in element_ids]
    ### get atom-wise lattice id
    coordinate_lattice_id = np.concatenate(
        [
            np.array([lattice_id] * num)
            for lattice_id, num in enumerate(optimized_mini_batch_inputs_dict["size"])
        ]
    )

    # save current poscar
    for lattice_id in range(len(size)):
        atom_dic = {
            "lattice_mat": [list(vec) for vec in lattice_vectors[lattice_id]],
            "coords": [
                list(vec)
                for vec in normed_batch_dir_coords[coordinate_lattice_id == lattice_id]
            ],
            "elements": np.array(elements)[
                coordinate_lattice_id == lattice_id
            ].tolist(),
            "abc": batch_abc[lattice_id].tolist(),
            "angles": batch_angle[lattice_id].tolist(),
            "cartesian": False,
            "props": [""] * len(batch_dir_coords[coordinate_lattice_id == lattice_id]),
        }
        ### if nan exists, use initial poscar. Otherwise, use optimized poscar
        if (
            np.isnan(lattice_vectors[lattice_id]).any()
            or np.isnan(batch_abc[lattice_id]).any()
            or np.isnan(batch_angle[lattice_id]).any()
            or np.isnan(atomic_distribution[coordinate_lattice_id == lattice_id]).any()
            or np.isnan(batch_dir_coords[coordinate_lattice_id == lattice_id]).any()
        ):
            print(f"lattice_id: {fnames[lattice_id]} has nan. use initial poscar")
            shutil.copy(
                f"./init_poscars/{fnames[lattice_id]}",
                os.path.join(tmp_poscar_dir, f"{fnames[lattice_id]}"),
            )
        else:
            Atoms.from_dict(atom_dic).write_poscar(
                os.path.join(tmp_poscar_dir, f"{fnames[lattice_id]}")
            )
