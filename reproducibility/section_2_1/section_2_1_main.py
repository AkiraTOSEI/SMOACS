#!/usr/bin/env python
# coding: utf-8

# # mutaionや初期値選定の効果を確かめるノートブック
# - 完全ランダム初期値と比較する
# - 共通の初期値データセットから最適化を行う

# In[ ]:


import json
import os
import random
import sys
from pathlib import Path

SECTION_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("SMOACS_ROOT", SECTION_DIR.parents[1])).resolve()
if not REPO_ROOT.is_dir():
    raise ValueError(f"Repository root does not exist: {REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
from tqdm import tqdm


# In[ ]:


try:
    from IPython.display import clear_output, display

    # 使用可能な環境の場合の処理
    def safe_display(*args, **kwargs):
        display(*args, **kwargs)

    def safe_clear_output(*args, **kwargs):
        clear_output(*args, **kwargs)

except ImportError:
    # IPython関連の関数が使用不可能な環境の場合の処理
    def safe_display(*args, **kwargs):
        if args:
            print(*args)
        if kwargs.get("display"):
            print(kwargs["display"])

    def safe_clear_output(*args, **kwargs):
        pass  # 何もしないか、他の何らかのクリア処理を実行


# ## 一般的に使う関数

# In[ ]:


def get_master_dir():
    return str(REPO_ROOT)


# ## データロード

# In[ ]:


import argparse
import glob
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# In[ ]:
from jarvis.core.specie import get_node_attributes
from jarvis.db.figshare import data

# ### データの読み込み

# ### ALIGNN

# #### あらかじめMegNetデータの保存したデータから読み込む

# In[ ]:


def create_ALIGNN_init_structure_from_directory(
    dir_path: str,
    intermid_dir_path: str,
    max_ox_states: int,
    mask_method: str,
    crystal_system: Optional[str] = None,
    num_candidate: Optional[int] = None,
    graph_update_mode: bool = False,
):
    print("<info> load initial structures from", dir_path)
    if crystal_system is None:
        print("<info> crystal_system is None")
        mask_d = dict(
            np.load(
                os.path.join(intermid_dir_path, f"mask_dict_{mask_method}.npz"),
                allow_pickle=True,
            )
        )
        radii_d = None
    elif crystal_system == "perovskite":
        print("<info> crystal_system is perovskite")
        mask_d = dict(
            np.load(
                os.path.join(intermid_dir_path, f"ionic_mask_dict_{mask_method}.npz"),
                allow_pickle=True,
            )
        )
        radii_d = dict(
            np.load(
                os.path.join(intermid_dir_path, f"ionic_radii_dict_{mask_method}.npz"),
                allow_pickle=True,
            )
        )

    dataset_optimize = []
    num_total_crystals = 0
    file_paths = glob.glob(os.path.join(dir_path, "*.vasp"))
    for path in file_paths:
        atoms = Atoms.from_poscar(path)

        (
            atom_mask_for_all_ox_states,
            radii_for_all_ox_states,
            ox_states_used_mask,
            site_ids,
        ) = get_atom_masks_for_oxidation_states(
            atoms=atoms,
            mask_d=mask_d,
            radii_d=radii_d,
            max_ox_states=max_ox_states,
            crystal_system=crystal_system,
            graph_update_mode=graph_update_mode,
        )
        dataset_optimize.append(
            {
                "jid": os.path.basename(path),
                "atoms": {
                    "lattice_mat": atoms.lattice.matrix,
                    "coords": atoms.coords,
                    "elements": atoms.elements,
                    "abc": atoms.lattice.abc,
                    "angles": atoms.lattice.angles,
                    "props": atoms.props,
                    "cartesian": False,
                },
                "dummy_target": 0,  # 0 はダミー
                "elements": atoms.elements,
                "atom_mask_for_all_ox_states": atom_mask_for_all_ox_states,
                "radii_for_all_ox_states": radii_for_all_ox_states,
                "ox_states_used_mask": ox_states_used_mask,
                "site_ids": site_ids,
            }
        )

    if num_candidate is not None:
        dataset_optimize = dataset_optimize[:num_candidate]

    num_total_crystals = len(dataset_optimize)
    return dataset_optimize, num_total_crystals


# #### 電気的な中性をたもつ関数

# In[ ]:


# In[ ]:
# In[ ]:
import itertools

from jarvis.core.atoms import Atoms
from pymatgen.core.periodic_table import Element as Element_pmg
from smact import Element, element_dictionary, neutral_ratios


def elec_neutral_all(
    elements: List[str], stoichs: List[List[int]], max_ox_states: int
) -> Tuple[bool, Optional[Tuple[int]]]:
    """
    Check for electrical neutrality using SMACT by evaluating possible oxidation states combinations.

    Args:
        elements (List[str]): List of element symbols.
        stoichs (List[List[int]]): List of lists containing stoichiometries.

    Returns:
        Tuple[bool, Optional[Tuple[int]]]: A tuple where the first element is a boolean indicating
                                           whether the input is electrically neutral, and the second
                                           element is a tuple of oxidation states that make it neutral
                                           (or None if no neutral combination is found).

    Examples:
        >>> elec_neutral_check_SMACT(elements=['Ti', 'O'], stoichs=[[1], [2]])
        [(2, -1), (4, -2)]
        >>> elec_neutral_check_SMACT(elements = ['Ti', 'Al', 'O'], stoichs = [[1],[1],[1]])
        [(-1, 3, -2)]
        >>> elec_neutral_all(elements = ['Cs', 'Pb', 'I'], stoichs = [[1],[1],[3]])
        [(1, -4, 1), (1, 2, -1)]
        >>> elec_neutral_all(elements = ['Ba', 'Ti', 'O'], stoichs = [[1],[1],[3]])
        [(2, 4, -2)]
    """
    space = element_dictionary(elements)
    els = [space[element] for element in elements]
    ox_combos = [el.oxidation_states for el in els]

    neutral_ox_states = []
    for ox_states in itertools.product(*ox_combos):
        cn_e, cn_r = neutral_ratios(ox_states, stoichs=stoichs, threshold=None)
        if cn_e:
            # PyMatGenにその酸化数が含まれるかどうかを判定する
            ox_in_pmg = True
            for ox, atom in zip(ox_states, elements):
                ox_in_pmg = ox_in_pmg and (ox in Element_pmg(atom).oxidation_states)
            if ox_in_pmg:
                neutral_ox_states.append(ox_states)
                if len(neutral_ox_states) == max_ox_states:
                    return neutral_ox_states
    return neutral_ox_states


# In[ ]:


def elec_neutral_all_common(
    elements: List[str], stoichs: List[List[int]], max_ox_states: int
) -> List[Tuple[int]]:
    """
    Determine all possible combinations of oxidation states for a list of elements that result in electrical neutrality.

    Args:
        elements (List[str]): A list of element symbols.
        stoichs (List[List[int]]): A list of stoichiometries corresponding to each element. Each sublist should contain exactly one integer.
        max_ox_states (int): The maximum number of oxidation state combinations to return.

    Returns:
        List[Tuple[int]]: A list of tuples, where each tuple contains a combination of oxidation states that sum to zero.

     Examples:
        >>> elec_neutral_all_common(['Ti', 'O'], [[1], [2]])
        [(4, -2, -2)]
        >>> elec_neutral_all_common(['Fe', 'O'], [[1], [1]])
        [(2, -2)]
        >>> elec_neutral_all_common(['H', 'O'], [[1], [2]])
        []
    """
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem] * stoi[0])
    ox_combos = [
        list(
            set(Element_pmg(elem).icsd_oxidation_states)
            & set(Element_pmg(elem).oxidation_states)
            & set(Element(elem).oxidation_states)
        )
        for elem in all_elements
    ]

    all_ox_states = []
    for ox_states in itertools.product(*ox_combos):
        if sum(ox_states) == 0:
            all_ox_states.append(ox_states)
        if len(all_ox_states) == max_ox_states:
            return all_ox_states, all_elements
    return all_ox_states, all_elements


# In[ ]:


# In[ ]:


def find_positions(lst: List[str], element: str) -> List[int]:
    """Find all positions of an element in a list."""
    return [index for index, value in enumerate(lst) if value == element]


def get_atom_masks_for_oxidation_states(
    atoms: Atoms,
    mask_d: Dict[str, np.ndarray],
    max_ox_states: int,
    crystal_system: Optional[str],
    graph_update_mode: bool = False,
    radii_d: Optional[Dict[str, np.ndarray]] = None,
    num_type_of_atoms: int = 98,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate atom masks for all possible oxidation states combinations ensuring electrical neutrality, specific to the crystal system provided.

    Args:
        atoms (jarvis.core.atoms.Atoms): An object containing element information.
        mask_d (Dict[str, np.ndarray]): Dictionary mapping oxidation states (str) to masks (np.ndarray).
        max_ox_states (int): Maximum number of oxidation states to consider.
        crystal_system (Optional[str]): Type of crystal system, e.g., 'perovskite' or None.
        radii_d (Optional[Dict[str, np.ndarray]]): Dictionary of radii associated with elements, required for certain crystal systems.
        num_type_of_atoms (int): Total number of unique atom types considered.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - torch.Tensor: A 3D tensor where each slice along the third axis represents the atom masks for a particular combination of oxidation states. Shape is [num_atoms, num_type_of_atoms, max_ox_states].
            - torch.Tensor: A 3D tensor corresponding to radii for each element based on oxidation states. Shape is [num_atoms, num_type_of_atoms, max_ox_states].
            - torch.Tensor: A 1D tensor indicating which oxidation states were used (1) or not used (0) across combinations. Shape is [max_ox_states].
    """
    # Get neutral oxidation states
    if graph_update_mode:
        # graph updadeのときは、原子マスクを引き継ぐので、Noneにする
        assert (
            crystal_system is None
        ), "crystal_system must be None for graph update mode."
        site_ids = [np.nan] * len(atoms.elements)
        neutral_ox_states = []

    elif crystal_system == "perovskite":
        # Ti(4+), Ba(2+), O(2-)
        # print("perovskite_test",(elem_list, stoichs))
        p1_list, p2_list = [], []
        site_ids = []
        for elem in atoms.elements:
            if elem == "Ti":
                p1_list.append(2)
                p2_list.append(4)
                site_ids.append(1)  # B-site
            elif elem == "Ba":
                p1_list.append(1)
                p2_list.append(2)
                site_ids.append(0)  # A-site
            elif elem == "O":
                p1_list.append(-1)
                p2_list.append(-2)
                site_ids.append(2)  # O-site
            else:
                raise ValueError("Invalid element for perovskite crystal mode.")
        neutral_ox_states = [tuple(p1_list), tuple(p2_list)]
        assert (
            radii_d is not None
        ), "Radii mask dictionary is required for perovskite crystal mode."
    elif crystal_system is None:
        elem_list = list(set(atoms.elements))
        stoichs = [[atoms.elements.count(elem)] for elem in elem_list]
        neutral_ox_states, all_elements = elec_neutral_all_common(
            elem_list, stoichs, max_ox_states=max_ox_states
        )
        assert (
            len(neutral_ox_states) > 0
        ), "No neutral oxidation states found."  # あとで消す
        site_ids = [np.nan] * len(atoms.elements)
    else:
        raise ValueError("Invalid crystal_system. Choose from ['perovskite', None].")

    # Add dummy tuples until the length is max_ox_states
    dummy_tuple = tuple([-999] * len(atoms.elements))
    while len(neutral_ox_states) < max_ox_states:
        neutral_ox_states.append(dummy_tuple)

    atom_mask_for_all_ox_states = []
    radii_tensor_for_all_ox_states = []
    ox_states_used_mask = []
    for ox_states in neutral_ox_states:
        # Initialize mask for each oxidation state
        ox_mask = np.zeros((len(atoms.elements), num_type_of_atoms))
        radii_tensor = np.zeros((len(atoms.elements), num_type_of_atoms))

        # Skip dummy oxidation states
        if ox_states == dummy_tuple:
            ox_states_used_mask.append(0)
            atom_mask_for_all_ox_states.append(ox_mask)
            radii_tensor_for_all_ox_states.append(radii_tensor)
            continue
        else:
            ox_states_used_mask.append(1)

        #  Apply masks for each element based on oxidation states
        for pos, ox_val in enumerate(ox_states):
            if str(ox_val) not in mask_d:
                raise ValueError(
                    f"Oxidation state {ox_val} not found in mask dictionary. Element: {all_elements[pos]}"
                )
            ox_mask[pos] = mask_d[str(ox_val)]
            if radii_d is not None:
                radii_tensor[pos] = radii_d[str(ox_val)]

        atom_mask_for_all_ox_states.append(ox_mask)
        radii_tensor_for_all_ox_states.append(radii_tensor)

    return (
        torch.tensor(np.stack(atom_mask_for_all_ox_states, axis=-1)),
        torch.tensor(np.stack(radii_tensor_for_all_ox_states, axis=-1)),
        torch.tensor(ox_states_used_mask),
        torch.tensor(site_ids),
    )


# In[ ]:


# In[ ]:


# #### ランダムなペロブスカイト構造を初期値にする

# In[ ]:


from pymatgen.core.periodic_table import Element as Element_pmg


def show_tolerance_factors():
    """
    ペロブスカイトの許容因子（tolerance factor）の計算値を例示する関数
    """
    r_a = Element_pmg("Ba").ionic_radii[2]
    r_b = Element_pmg("Ti").ionic_radii[4]
    r_o = Element_pmg("O").ionic_radii[-2]
    t = (r_a + r_o) / ((2**0.5) * (r_b + r_o))
    print(f"tolerance factor for BaTiO3: {t:.3f}")
    r_a = Element_pmg("Ba").ionic_radii[2]
    r_b = Element_pmg("Ce").ionic_radii[4]
    r_o = Element_pmg("O").ionic_radii[-2]
    t = (r_a + r_o) / ((2**0.5) * (r_b + r_o))
    print(f"tolerance factor for BaCeO3: {t:.3f}")
    r_a = Element_pmg("Sr").ionic_radii[2]
    r_b = Element_pmg("Ti").ionic_radii[4]
    r_o = Element_pmg("O").ionic_radii[-2]
    t = (r_a + r_o) / ((2**0.5) * (r_b + r_o))
    print(f"tolerance factor for SrTiO3: {t:.3f}")


# ### モデルのデータ前処理

# #### ALIGNNのデータ前処理関数

# In[ ]:


def TEST__compute_direct_to_cartesian(
    train_df: pd.DataFrame, bondlegth_gt: torch.Tensor
) -> None:
    """
    バッチ内部に存在する原子の座標を直交座標に変換する関数をテストする。
    Args:
        train_df: Pytorch Dataset作成時に同時作成した使用する結晶構造の情報を格納したデータフレーム
        bondlength_gt: テストデータとして用いるbondlengthの正解データ

    """
    # POSCARのデータからbondlengthを再構成する
    (
        batch_dir_coords,
        batch_dst_ids,
        batch_src_ids,
        batch_displace,
        lattice_vectors,
        num_atoms,
        num_edges,
    ) = get_all_atoms_coords(train_df)

    batch_lattice_vectors = create_batch_lattice_vectors(lattice_vectors, num_edges)
    # test for batch_lattice_vectors
    previous_i = 0
    for i, num in enumerate(num_edges):
        test_lattice_vectors = batch_lattice_vectors[
            previous_i : previous_i + num_edges[i]
        ]
        assert (
            ((test_lattice_vectors - test_lattice_vectors[0:1]) == 0).numpy()
        ).all()  # 同じ結晶に所属しているはずなので、すべてのlattice_vectorsが一致するはず
        torch.testing.assert_close(
            test_lattice_vectors[0], lattice_vectors[i]
        )  # i番目のlattice_vectorsと結晶ごとのlattice_vectorsが一致するはず
        previous_i += num_edges[i]

    _, batch_bondlength = compute_bondlength(
        batch_dir_coords,
        batch_dst_ids,
        batch_src_ids,
        batch_displace,
        batch_lattice_vectors,
    )
    # bondlengthの正解データと一致するかを確認する
    torch.testing.assert_close(batch_bondlength, bondlegth_gt)


# In[ ]:


import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms

from third_party.alignn_minimal.graphs import Graph, StructureDataset


def load_graphs(
    dataset=[],
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    cutoff_extra: float = 3,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = None,
    use_canonize: bool = False,
    id_tag="jid",
    # extra_feats_json=None,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            cutoff_extra=cutoff_extra,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            neighbor_strategy=neighbor_strategy,
        )

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        # print('dataset',dataset,type(dataset))
        print("Converting to graphs!")
        graphs = []
        # columns=dataset.columns
        for ii, i in tqdm(dataset.iterrows(), total=len(dataset)):
            # print('iooooo',i)

            atoms = i["atoms"]
            # atomsにNaNを含まれている場合は、init_poscarのグラフを読み込む
            if (
                np.isnan(atoms["lattice_mat"]).any()
                or np.isnan(atoms["coords"]).any()
                or np.isnan(atoms["abc"]).any()
                or np.isnan(atoms["angles"]).any()
            ):
                atoms = Atoms.from_poscar(f"init_poscars/{i['jid']}").to_dict()
                print(
                    f"nan included in {i['jid']}, so load from init_poscars to convert to graph"
                )

            structure = Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
            g = Graph.atom_dgl_multigraph(
                structure,
                cutoff=cutoff,
                cutoff_extra=cutoff_extra,
                atom_features="atomic_number",
                max_neighbors=max_neighbors,
                compute_line_graph=False,
                use_canonize=use_canonize,
                neighbor_strategy=neighbor_strategy,
                id=i[id_tag],
            )
            atom_mask_for_all_ox_states = i["atom_mask_for_all_ox_states"]
            ox_states_used_mask = i["ox_states_used_mask"]
            radii_for_all_ox_states = i["radii_for_all_ox_states"]
            g.ndata["atom_mask_for_all_ox_states"] = torch.tensor(
                atom_mask_for_all_ox_states
            ).type(torch.get_default_dtype())
            g.ndata["ox_states_used_mask"] = torch.tensor(
                torch.stack([ox_states_used_mask] * len(atom_mask_for_all_ox_states))
            ).type(torch.get_default_dtype())
            g.ndata["radii_for_all_ox_states"] = torch.tensor(
                radii_for_all_ox_states
            ).type(torch.get_default_dtype())
            g.ndata["crystal_id"] = torch.tensor(
                [ii] * len(atom_mask_for_all_ox_states), dtype=torch.int64
            )
            g.ndata["site_ids"] = i["site_ids"]
            # print(i["atoms"]["elements"])
            # g.ndata['original_elements'] = torch.tensor(i["atoms"]["elements"])
            if "extra_features" in i:
                natoms = len(atoms["elements"])
                # if "extra_features" in columns:
                g.ndata["extra_features"] = torch.tensor(
                    [i["extra_features"] for n in range(natoms)]
                ).type(torch.get_default_dtype())
            graphs.append(g)

        # df = pd.DataFrame(dataset)
        # print ('df',df)

        # graphs = df["atoms"].progress_apply(atoms_to_graph).values
        # print ('graphs',graphs,graphs[0])
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs


# In[ ]:


import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def get_torch_dataset(
    dataset=[],
    id_tag="jid",
    target="",
    target_atomwise="",
    target_grad="",
    target_stress="",
    neighbor_strategy="",
    atom_features="",
    use_canonize="",
    name="",
    line_graph="",
    cutoff=8.0,
    cutoff_extra=3.0,
    max_neighbors=12,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    # df['natoms']=df['atoms'].apply(lambda x: len(x['elements']))
    # print(" data df", df)
    vals = np.array([ii[target] for ii in dataset])  # df[target].values
    print("data range", np.max(vals), np.min(vals))
    f = open(os.path.join(output_dir, tmp_name + "_data_range"), "w")
    line = "Max=" + str(np.max(vals)) + "\n"
    f.write(line)
    line = "Min=" + str(np.min(vals)) + "\n"
    f.write(line)
    f.close()

    graphs = load_graphs(
        df,
        name=name,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        cutoff_extra=cutoff_extra,
        max_neighbors=max_neighbors,
        id_tag=id_tag,
    )

    print("StructureDatasetに入力する変数")
    print("target=", target)
    print("target_atomwise=", target_atomwise, type(target_atomwise))
    print("target_grad=", target_grad, type(target_grad))
    print("target_stress=", target_stress, type(target_stress))
    print("atom_features=", atom_features, type(atom_features))
    print("line_graph=", line_graph, type(line_graph))
    print("id_tag=", id_tag)
    print("classification=", classification)

    st_data = StructureDataset(
        df[["jid", "atoms", "dummy_target"]],
        graphs,
        target=target,
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )
    return st_data, df


# In[ ]:


def get_all_atoms_coords(train_df: pd.DataFrame) -> Tuple[torch.Tensor]:
    """
    train_dfをもとに、使用する結晶構造データから全原子のdirect coordinateと、結合のsrcとdistを算出する。
    Args:
        train_df: Pytorch Dataset作成時に同時作成した使用する結晶構造の情報を格納したデータフレーム
    Outputs:
        batch_dir_coords: 全原子のdirect coordinateをまとめたtensor
        batch_dst_ids: 全結合の終点のIDをまとめたtensor
        batch_src_ids: 全結合の始点のIDをまとめたtensor
        batch_displace: 全結合のdisplacementをまとめたtensor
        batch_lattice_vectors: 全結晶構造のlattice vectorsをまとめたtensor
    """
    _dir_coords, _lattice_vectors = [], []
    _dst_ids, _src_ids, _displace = [], [], []
    num_atoms, num_edges = [], []
    num_total_atoms = 0
    for atoms_dict in train_df.atoms:
        atoms = Atoms.from_dict(atoms_dict)
        _dir_coords.append(torch.Tensor(atoms.frac_coords))
        edges = nearest_neighbor_edges(atoms, use_canonize=True)
        y_src_ids, y_dst_ids, y_displace = build_undirected_edge_data_tensor(edges)
        lattice_vectors = torch.Tensor(atoms.lattice_mat)
        # _lattice_vectors.append(torch.stack([lattice_vectors]*len(y_dst_ids)))
        _lattice_vectors.append(lattice_vectors.unsqueeze(0))
        _dst_ids.append(y_dst_ids + num_total_atoms)
        _src_ids.append(y_src_ids + num_total_atoms)
        _displace.append(y_displace)
        num_total_atoms += len(atoms.frac_coords)
        num_atoms.append(len(atoms.frac_coords))
        num_edges.append(len(y_dst_ids))

    batch_dir_coords = torch.cat(_dir_coords)
    batch_dst_ids = torch.cat(_dst_ids)
    batch_src_ids = torch.cat(_src_ids)
    batch_displace = torch.cat(_displace)
    batch_lattice_vectors = torch.cat(_lattice_vectors)

    return (
        batch_dir_coords,
        batch_dst_ids,
        batch_src_ids,
        batch_displace,
        batch_lattice_vectors,
        num_atoms,
        num_edges,
    )


# In[ ]:


def generate_feature_matrix() -> np.ndarray:
    """
    Generate a matrix of features for each atomic element in the periodic table.

    This function creates a feature matrix where each row corresponds to an atom
    and each column corresponds to a particular feature of that atom, such as
    atomic number, electronegativity, etc.

    Returns:
        A numpy ndarray where each row is an array of features for a single atom.
    """
    atom_list = create_element_list()
    feats = []
    for i, elem in enumerate(atom_list):
        feats.append(get_node_attributes(elem, atom_features="cgcnn"))
    atom_feat_matrix = torch.Tensor(np.array(feats))
    return atom_feat_matrix


# In[ ]:


def create_learnable_oxides_mask(mini_batch_inputs_dict):
    """
    a function to create learnable mask tensor for oxides used in the model
    """
    _ox_mask_learnable_tensor_per_crystal = []
    total_idx = 0

    for idx in mini_batch_inputs_dict["size"]:
        total_idx += idx
        _ox_mask_learnable_tensor_per_crystal.append(
            mini_batch_inputs_dict["ox_states_used_mask"][total_idx - 1]
        )
    # ox_mask_learnable_tensor_per_crystal = torch.nn.Parameter(torch.stack(_ox_mask_learnable_tensor_per_crystal))
    ox_mask_learnable_tensor_per_crystal = torch.stack(
        _ox_mask_learnable_tensor_per_crystal
    )

    # test
    stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
        ox_mask_learnable_tensor_per_crystal, mini_batch_inputs_dict["size"]
    )
    # print("stacked_learnable_ox_weight", stacked_learnable_ox_weight, stacked_learnable_ox_weight.shape)
    # print("ox_states_used_mask", mini_batch_inputs_dict['ox_states_used_mask'], mini_batch_inputs_dict['ox_states_used_mask'].shape)
    assert (
        (
            stacked_learnable_ox_weight.detach()
            == mini_batch_inputs_dict["ox_states_used_mask"]
        )
        .all()
        .detach()
        .cpu()
        .numpy()
    )

    return ox_mask_learnable_tensor_per_crystal


# In[ ]:


def stacking_learnable_oxsides_mask(
    ox_mask_learnable_tensor_per_crystal: torch.Tensor, size: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        ox_mask_learnable_tensor_per_crystal: torch.Tensor, shape=(num_crystals, num_oxides)
    """
    return torch.concat(
        [
            torch.stack([ox_mask_learnable_tensor_per_crystal[idx]] * size)
            for idx, size in enumerate(size)
        ],
        dim=0,
    )


# In[ ]:


# In[ ]:


def graph_to_tensors(
    g: dgl.DGLGraph,
    lg: dgl.DGLGraph,
    train_df: pd.DataFrame,
    data_dict_for_test: Dict[str, torch.Tensor],
    device: str = "cuda",
    use_atomic_mask: bool = False,
    test_mode: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Graphデータから、学習に使用するtensorデータを生成する。

    """
    ### Atomic Featuresの初期化
    # 原子ごとの特徴量と、原子分布から、分布化された特徴量を生成する
    # 学習可能なものは原子分布
    atom_feat_matrix = generate_feature_matrix()

    # 初期化。学習可能な原子分布を一様分布として生成する
    atomic_dist = torch.ones(
        np.sum(train_df["elements"].apply(len).values), atom_feat_matrix.shape[0]
    )
    atomic_dist = atomic_dist / atomic_dist.sum(1).view(-1, 1)

    # initialization for atomic masks
    atom_mask_for_all_ox_states = g.ndata.pop("atom_mask_for_all_ox_states")
    ox_states_used_mask = g.ndata.pop("ox_states_used_mask")
    radii_for_all_ox_states = g.ndata.pop("radii_for_all_ox_states")
    site_ids = g.ndata.pop("site_ids")
    # original_elements = g.ndata.pop('original_elements')

    # initialization for direct coordiantes, lattice vectors, src_ids, dst_ids, displace
    (
        batch_dir_coords,
        batch_dst_ids,
        batch_src_ids,
        batch_displace,
        lattice_vectors,
        num_atoms,
        num_edges,
    ) = get_all_atoms_coords(train_df)

    # initialization for lattice vectors: a,b,c,alpha,beta,gamma
    batch_abc, batch_angle = compute_abc_angle(lattice_vectors)

    # initialization for angles
    z_src_ids, z_dst_ids = lg.edges()

    # count the number of atoms in each crystal
    _size = []
    for crystal_id in g.ndata["crystal_id"].unique():
        _size.append(g.ndata["crystal_id"].eq(crystal_id).sum().item())
    size = torch.tensor(_size).long()

    if test_mode:
        # test for atomic features
        # atom_feat_matrix.shape = (num_atom_species(98), num_features(92)), atomic_dist.shape = (num_atoms, num_atom_species(98))
        init_atom_feat_gt = (
            torch.sum(atom_feat_matrix, dim=0) / atom_feat_matrix.shape[0]
        )
        atomic_features4test = torch.matmul(atomic_dist, atom_feat_matrix)
        torch.testing.assert_close(atomic_features4test[0], init_atom_feat_gt)
        # test for bond length
        torch.testing.assert_close(g.edges()[0], batch_src_ids)
        torch.testing.assert_close(g.edges()[1], batch_dst_ids)
        lattice_vectors4test = compute_lattice_vectors(batch_abc, batch_angle)
        batch_lattice_vectors4test = create_batch_lattice_vectors(
            lattice_vectors4test, num_edges
        )
        edge_vectors4test, bondlength4test = compute_bondlength(
            batch_dir_coords,
            batch_dst_ids,
            batch_src_ids,
            batch_displace,
            batch_lattice_vectors4test,
        )
        torch.testing.assert_close(
            bondlength4test, data_dict_for_test["bondlength"], atol=3e-5, rtol=5e-6
        )
        TEST__compute_direct_to_cartesian(train_df, data_dict_for_test["bondlength"])
        # test for bond angles
        lg.apply_edges(compute_bond_cosines_for_line_graph)
        bond_angles4test = calculate_bond_cosine(
            -edge_vectors4test[z_src_ids], edge_vectors4test[z_dst_ids]
        )
        torch.testing.assert_close(
            bond_angles4test, data_dict_for_test["angle_features"], atol=3e-5, rtol=5e-6
        )
        torch.testing.assert_close(
            bond_angles4test, lg.edata["h"], atol=3e-5, rtol=5e-6
        )

    ### to GPU
    batch_dst_ids = batch_dst_ids  # .to(device)
    batch_src_ids = batch_src_ids  # .to(device)
    z_src_ids = z_src_ids  # .to(device)
    z_dst_ids = z_dst_ids  # .to(device)
    batch_displace = batch_displace  # .to(device)
    num_edges = torch.tensor(num_edges)  # .to(device)
    atom_feat_matrix = atom_feat_matrix  # .to(device)
    # g = g.to(device)
    # lg = lg.to(device)

    alignn_minibatch_dict = {
        "batch_dir_coords": batch_dir_coords,  # .to(device),
        "batch_abc": batch_abc,  # .to(device),
        "batch_angle": batch_angle,  # .to(device),
        "init_coords": batch_dir_coords.clone(),  # .to(device),
        "init_abc": batch_abc.clone(),  # .to(device),
        "init_angles": batch_angle.clone(),  # .to(device),
        "atomic_distribution": atomic_dist,  # .to(device),
        "batch_dst_ids": batch_dst_ids,
        "batch_src_ids": batch_src_ids,
        "z_src_ids": z_src_ids,
        "z_dst_ids": z_dst_ids,
        "size": size,
        "site_ids": site_ids,
        "batch_displace": batch_displace,
        "num_edges": num_edges,
        "atom_feat_matrix": atom_feat_matrix,
        "g": g,
        "lg": lg,
        "ox_states_used_mask": ox_states_used_mask,  # .to(device),
        #'original_elements':original_elements,
        "fnames": train_df["jid"].values,
        "atomic_mask": None,
        "radii_tensor": None,
        "ox_mask_learnable_tensor_per_crystal": None,
    }

    if use_atomic_mask:
        alignn_minibatch_dict["atomic_mask"] = (
            atom_mask_for_all_ox_states  # .to(device)
        )
        ox_mask_learnable_tensor_per_crystal = create_learnable_oxides_mask(
            alignn_minibatch_dict
        )
        alignn_minibatch_dict["ox_mask_learnable_tensor_per_crystal"] = (
            ox_mask_learnable_tensor_per_crystal
        )
        alignn_minibatch_dict["radii_tensor"] = radii_for_all_ox_states  # .to(device)

    return alignn_minibatch_dict


# In[ ]:


from collections import defaultdict


def nearest_neighbor_edges(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):
    """Construct k-NN edge list."""
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    # print("len(all_neighbors):",len(all_neighbors))

    # if a site has too few neighbors, increase the cutoff radius
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)
    # print("min_nbrs:",min_nbrs)
    # print("use_canonize:",use_canonize)
    attempt = 0
    # print ('cutoff=',all_neighbors)
    if min_nbrs < max_neighbors:
        # print("extending cutoff radius!", attempt, cutoff, id)
        lat = atoms.lattice
        if cutoff < max(lat.a, lat.b, lat.c):
            r_cut = max(lat.a, lat.b, lat.c)
        else:
            r_cut = 2 * cutoff
        attempt += 1

        return nearest_neighbor_edges(
            atoms=atoms,
            use_canonize=use_canonize,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            id=id,
        )
    # build up edge list
    # NOTE: currently there's no guarantee that this creates undirected graphs
    # An undirected solution would build the full edge list where nodes are
    # keyed by (index, image), and ensure each edge has a complementary edge

    # indeed, JVASP-59628 is an example of a calculation where this produces
    # a graph where one site has no incident edges!

    # build an edge dictionary u -> v
    # so later we can run through the dictionary
    # and remove all pairs of edges
    # so what's left is the odd ones out
    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        # sort on distance
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        # find the distance to the k-th nearest neighbor
        max_dist = distances[max_neighbors - 1]
        # max_dist = distances[max_neighbors - 1]

        # keep all edges out to the neighbor shell of the k-th neighbor
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]

        # keep track of cell-resolved edges
        # to enforce undirected graph construction
        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges


def build_undirected_edgedata(
    atoms=None,
    edges={},
):
    """Build undirected graph data from edge set.

    edges: dictionary mapping (src_id, dst_id) to set of dst_image
    r: cartesian displacement vector from src -> dst
    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(dst_coord - atoms.frac_coords[src_id])
            # if np.linalg.norm(d)!=0:
            # print ('jv',dst_image,d)
            # add edges for both directions
            for uu, vv, dd in [(src_id, dst_id, d), (dst_id, src_id, -d)]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
    u, v, r = (np.array(x) for x in (u, v, r))
    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())

    return u, v, r


# In[ ]:


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image
    """
    # store directed edges src_id <= dst_id
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    # shift periodic images so that src is in (0,0,0) image
    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    assert src_image == (0, 0, 0)

    return src_id, dst_id, src_image, dst_image


# In[ ]:


from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


def build_undirected_edge_data_tensor(
    edges: defaultdict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create tensors for edge information of atoms in a crystal.

    Based on the build_undirected_edgedata function in alignn/graphs.py, this function constructs
    tensors that represent source nodes, destination nodes, and displacement vectors for each edge
    in an undirected graph representation of a crystal structure.

    Args:
        edges (defaultdict): A dictionary with keys as tuples of atom indices (source, destination)
                             and values as lists of image displacement vectors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Three tensors containing the source ids,
                                                          destination ids, and displacement vectors
                                                          for each edge in the graph.

    Note:
        The function assumes the edge dictionary contains information for an undirected graph, hence
        for each edge from src_id to dst_id, an opposite edge from dst_id to src_id is added with
        the displacement negated.
    """
    src, dst, displace = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            src.append(src_id)
            dst.append(dst_id)
            displace.append(-1 * np.array(dst_image))

            # Adding the reverse edge for undirected graph
            src.append(dst_id)
            dst.append(src_id)
            displace.append(np.array(dst_image))

    # Convert lists to torch tensors
    src_ids = torch.tensor(src, dtype=torch.long)
    dst_ids = torch.tensor(dst, dtype=torch.long)
    displace = torch.tensor(displace, dtype=torch.float)

    return src_ids, dst_ids, displace


def direct_to_cartesian(
    dir_coords: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert direct coordinates to cartesian coordinates using lattice vectors.

    The conversion is done by matrix multiplying the direct coordinates with the lattice vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates, with shape (N, 3), where N is the number
                                   of atoms in the structure.
        lattice_vectors (torch.Tensor): The lattice vectors, with shape (3, 3).

    Returns:
        torch.Tensor: The cartesian coordinates, with the same shape as dir_coords.
    """
    return torch.matmul(dir_coords, lattice_vectors)


def direct_to_cartesian_batch(
    dir_coords: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert direct coordinates to cartesian coordinates using lattice vectors.

    The conversion is done by matrix multiplying the direct coordinates with the lattice vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates, with shape (N, 3), where N is the number
                                   of atoms in the structure.
        lattice_vectors (torch.Tensor): The lattice vectors, with shape (N, 3, 3).

    Returns:
        torch.Tensor: The cartesian coordinates, with the same shape as dir_coords.
    """
    _bool1 = len(dir_coords.shape) == 2
    _bool2 = len(lattice_vectors.shape) == 3
    _bool3 = dir_coords.shape[1] == 3
    if not (_bool1 and _bool2 and _bool3):
        print("dir_coords.shape:", dir_coords.shape)
        print("lattice_vectors.shape:", lattice_vectors.shape)
        raise Exception(
            "The shapes of dir_coords and lattice_vectors are not compatible."
        )

    cartesian_coords = torch.bmm(dir_coords.unsqueeze(1), lattice_vectors)
    return cartesian_coords.squeeze(1)


def compute_edge_vectors(
    dir_coords: torch.Tensor,
    lattice_vectors: torch.Tensor,
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    displace: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the edge vectors in cartesian coordinates for atoms in a crystal lattice based on their direct coordinates.

    This function computes the vectors connecting source and destination atoms considering the periodic boundary conditions
    represented by the displacement vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates of atoms in the crystal lattice, shape (N, 3), where N is the number of atoms.
        lattice_vectors (torch.Tensor): The lattice vectors defining the crystal lattice, shape (3, 3).
        src_ids (torch.Tensor): The indices of the source atoms for each edge, shape (edges_N,), where edges_N is the number of edges.
        dst_ids (torch.Tensor): The indices of the destination atoms for each edge, shape (edges_N,), where edges_N is the number of edges.
        displace (torch.Tensor): The translation vectors for periodic boundary conditions, shape (edges_N, 3), where edges_N is the number of edges.

    Returns:
        torch.Tensor: The vectors representing the edges in cartesian coordinates, shape (edges_N, 3), where edges_N is the number of edges.
    """
    # Calculate the difference in direct coordinates adjusted for displacement
    diff_dir_coords = dir_coords[dst_ids] - dir_coords[src_ids] - displace

    # Convert the difference in direct coordinates to cartesian coordinates
    edge_vectors = direct_to_cartesian_batch(diff_dir_coords, lattice_vectors)
    return edge_vectors


# In[ ]:


def compute_abc_angle(batch_lattice_vec: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Computes the side lengths and angles of crystal lattices from their lattice vectors for a batch.

    This function calculates the side lengths (a, b, c) and angles (alpha, beta, gamma) of the crystal lattices
    based on the provided lattice vectors for each item in the batch. The angles are computed in degrees.

    Args:
        batch_lattice_vec (torch.Tensor): A tensor containing the lattice vectors for the crystal lattices in the batch.
                                          The shape is (B, 3, 3), where B is the batch size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                           - The first tensor contains the side lengths (a, b, c) of the crystal lattices in the batch, with shape (B, 3).
                                           - The second tensor contains the angles (alpha, beta, gamma) in degrees of the crystal lattices in the batch, with shape (B, 3).
    """
    B = batch_lattice_vec.shape[0]
    batch_abc = torch.zeros(
        (B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device
    )
    batch_angle = torch.zeros(
        (B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device
    )

    # 辺の長さを計算
    batch_abc[:, 0] = torch.norm(batch_lattice_vec[:, 0, :], dim=1)
    batch_abc[:, 1] = torch.norm(batch_lattice_vec[:, 1, :], dim=1)
    batch_abc[:, 2] = torch.norm(batch_lattice_vec[:, 2, :], dim=1)

    # 成す角を計算
    # alpha = angle between b and c
    batch_angle[:, 0] = torch.acos(
        torch.sum(batch_lattice_vec[:, 1, :] * batch_lattice_vec[:, 2, :], dim=1)
        / (batch_abc[:, 1] * batch_abc[:, 2])
    )
    # beta = angle between a and c
    batch_angle[:, 1] = torch.acos(
        torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 2, :], dim=1)
        / (batch_abc[:, 0] * batch_abc[:, 2])
    )
    # gamma = angle between a and b
    batch_angle[:, 2] = torch.acos(
        torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 1, :], dim=1)
        / (batch_abc[:, 0] * batch_abc[:, 1])
    )

    # ラジアンから度へ変換
    batch_angle = torch.rad2deg(batch_angle)

    return batch_abc, batch_angle


# In[ ]:


import numpy as np  # numpyを使用してπを取得
import torch


def compute_lattice_vectors(
    batch_abc: torch.Tensor, batch_angle: torch.Tensor
) -> torch.Tensor:
    """
    Computes the lattice vectors for a batch of crystal lattices based on their side lengths and angles.

    This function calculates the lattice vectors for each set of crystal lattice parameters in the batch. The lattice
    vectors are computed using the side lengths (a, b, c) and angles (alpha, beta, gamma) provided for each crystal lattice.

    Args:
        batch_abc (torch.Tensor): A tensor containing the side lengths of the crystal lattices in the batch.
                                  The shape is (B, 3), where B is the batch size.
        batch_angle (torch.Tensor): A tensor containing the angles (in degrees) of the crystal lattices in the batch.
                                    The shape is (B, 3), where B is the batch size. Angles are in the order (alpha, beta, gamma).

    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) containing the lattice vectors for each crystal lattice in the batch.
    """
    # 角度をラジアンに変換
    angle_rad = batch_angle * np.pi / 180.0

    # 成す角のコサインとサイン
    cos_alpha = torch.cos(angle_rad[:, 0])
    cos_beta = torch.cos(angle_rad[:, 1])
    cos_gamma = torch.cos(angle_rad[:, 2])
    sin_gamma = torch.sin(angle_rad[:, 2])

    # バッチサイズの取得
    B = batch_abc.shape[0]

    # 格子ベクトルテンソルの初期化
    lattice_vec = torch.zeros((B, 3, 3), dtype=batch_abc.dtype, device=batch_abc.device)

    # 格子ベクトルの計算
    lattice_vec[:, 0, 0] = batch_abc[:, 0]  # a
    lattice_vec[:, 1, 0] = batch_abc[:, 1] * cos_gamma  # b*cos(gamma)
    lattice_vec[:, 1, 1] = batch_abc[:, 1] * sin_gamma  # b*sin(gamma)
    lattice_vec[:, 2, 0] = batch_abc[:, 2] * cos_beta  # c*cos(beta)
    lattice_vec[:, 2, 1] = (
        batch_abc[:, 2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    )
    lattice_vec[:, 2, 2] = batch_abc[:, 2] * torch.sqrt(
        1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma) ** 2
    )

    return lattice_vec


# In[ ]:


def create_batch_lattice_vectors(lattice_vectors: torch.Tensor, num_edges: List[int]):
    """
    lattice_vectors(num_candidates, 3, 3)をエッジ（結合）の数分だけ結晶ごとに拡張する関数
    """
    batch_lattice_vectors = torch.cat(
        [
            torch.cat([lattice_vectors[i : i + 1]] * num)
            for i, num in enumerate(num_edges)
        ]
    )
    return batch_lattice_vectors


# In[ ]:


### バッチ内部に存在する全原子の内部座標coordsと結晶ベクトルlattice_vectorsから、全原子のデカルト座標を計算し、bondlengthを再構成する関数を作成する
def compute_bondlength(
    coords: torch.Tensor,
    dst_ids: torch.Tensor,
    src_ids: torch.Tensor,
    displacement: torch.Tensor,
    lattice_vectors: torch.Tensor,
) -> Tuple[torch.Tensor]:
    """
    バッチ内部に存在する原子の座標を直交座標に変換する関数を作成する。
    Args:
        coords: バッチ内部に存在する全原子の内部座標
        dst_ids: バッチ内部に存在する全結合の終点のID
        src_ids: バッチ内部に存在する全結合の始点のID
        displacement: バッチ内部に存在する全結合のdisplacement
        lattice_vectors: バッチ内部に存在する全結晶構造のlattice vectors

    Returns:
        edge_vectors: バッチ内部に存在する全ての結合のデカルト座標
        bondlength: バッチ内部に存在する全ての結合のbondlength
    """

    edge_vectors = direct_to_cartesian_batch(
        coords[dst_ids] - coords[src_ids] - displacement, lattice_vectors
    )
    bondlength = torch.norm(edge_vectors, dim=1)

    return edge_vectors, bondlength


# In[ ]:


def compute_bond_cosines_for_line_graph(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    # bond_cosine = torch.arccos((torch.clamp(bond_cosine, -1, 1)))
    # print (r1,r1.shape)
    # print (r2,r2.shape)
    # print (bond_cosine,bond_cosine.shape)
    return {"h": bond_cosine}


# In[ ]:


def calculate_bond_cosine(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """calculate cosine
    Args:
        r1: shape=(N,3)
        r2: shape=(N,3)
    """
    bond_cosine = torch.nn.functional.cosine_similarity(r1, r2, dim=1)
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


# In[ ]:


import numpy as np


def load_ALIGNN_initial_data(
    settings_dict: Dict[str, Any],
    dir_path: str,
    intermid_dir_path: str,
    batch_size: int,
    use_atomic_mask: bool,
    max_ox_states: int,
    mask_method: str,
    crystal_system: Optional[str],
    angle_range: Tuple[float, float],
    abc_range: Tuple[float, float],
    num_candidate: Optional[int] = None,
    max_neighbors: int = 12,
    cutoff: float = 8.0,
    cutoff_extra: float = 3.0,
    base_lattice_index: int = 0,
    test_mode: bool = True,
    graph_update_mode: bool = False,
):
    """
    dft_3d のデータを読み込み、ランダムに選んだデータからグラフgとライングラフgl, 選んだデータの詳細train_dfwoを返す
    Args:
        batch_size(int): バッチサイズ
    Returns:
        g: グラフ
        lg: ライングラフ
        train_df: 選んだデータの詳細
        data_dict_for_test: 計算コードが正しいかをテストするための辞書
    """

    # intermid_dir_path = settings_dict['intermid_dir_path']
    # use_atomic_mask = settings_dict['use_atomic_mask']
    # max_ox_states = settings_dict['max_ox_states']
    # mask_method = settings_dict['mask_method']
    # crystal_system = settings_dict['crystal_system']
    # num_candidate = settings_dict['num_candidate']

    dataset_optimize, num_total = create_ALIGNN_init_structure_from_directory(
        dir_path=dir_path,
        crystal_system=crystal_system,
        intermid_dir_path=intermid_dir_path,
        max_ox_states=max_ox_states,
        num_candidate=num_candidate,
        mask_method=mask_method,
        graph_update_mode=graph_update_mode,
    )

    train_data, train_df = get_torch_dataset(
        dataset=dataset_optimize,
        id_tag="jid",
        atom_features="cgcnn",
        target="dummy_target",
        target_atomwise="",
        target_grad="",
        target_stress="",
        neighbor_strategy="k-nearest",
        use_canonize=True,
        name="dft_3d",
        line_graph=True,
        cutoff=cutoff,
        cutoff_extra=cutoff_extra,
        max_neighbors=max_neighbors,
        classification=False,
        output_dir="./",
        tmp_name="train_data",
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_data.collate_line_graph,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    ALIGNN_all_inputs = []
    for i, g_x in enumerate(train_loader):
        g = g_x[0]
        lg = g_x[1]

        g = g.local_var()
        lg = lg.local_var()
        # initial node features: atom feature network...
        atom_features = g.ndata.pop("atom_features")
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        angle_features = lg.edata.pop("h")

        data_dict_for_test = {
            "atom_features": atom_features,
            "bondlength": bondlength,
            "angle_features": angle_features,
        }

        alignn_minibatch_dict = graph_to_tensors(
            g=g,
            lg=lg,
            train_df=train_df.loc[i * batch_size : (i + 1) * batch_size - 1],
            data_dict_for_test=data_dict_for_test,
            device="cuda",
            use_atomic_mask=use_atomic_mask,
            test_mode=test_mode,
        )

        scalers = {
            "abc_scaler": ABC_Scaler(
                init_batch_abc=alignn_minibatch_dict["batch_abc"],
                min_length=abc_range[0],
                max_length=abc_range[1],
                device="cuda",
            ),
            "angle_scaler": Angle_Scaler(
                min_angle=angle_range[0],
                max_angle=angle_range[1],
            ),
        }

        ALIGNN_all_inputs.append([alignn_minibatch_dict, scalers])
        # if i*batch_size>=num_total:
        #    break

    return ALIGNN_all_inputs


# In[ ]:


# In[ ]:


# #### Crystalformerのデータ前処理関数

# In[ ]:


def load_crystalformer_initial_data(
    settings_dict: Dict[str, Any],
    dir_path: str,
    intermid_dir_path: str,
    crystal_system: Optional[str],
    batch_size: int,
    use_atomic_mask: bool,
    max_ox_states: int,
    mask_method: str,
    angle_range: Tuple[float, float],
    abc_range: Tuple[float, float],
    test_mode: bool,
    num_candidate: Optional[int] = None,
    device: str = "cuda",
):
    out_data = create_crystalformer_init_structure_from_directory(
        dir_path=dir_path,
        intermid_dir_path=intermid_dir_path,
        max_ox_states=max_ox_states,
        crystal_system=crystal_system,
        num_candidate=num_candidate,
        mask_method=mask_method,
    )

    (
        all_opt_abc,
        all_opt_x,
        all_batch,
        all_coords4input,
        all_size,
        all_angles,
        fnames,
        atom_mask_for_all_ox_states,
        radii_for_all_ox_states,
        ox_states_used_mask,
        site_ids,
    ) = out_data
    crystalformer_all_inputs_dict = {
        "x": all_opt_x,
        "batch": all_batch,
        "coords4input": all_coords4input,
        "size": all_size,
        "abc": all_opt_abc,
        "angles": all_angles,
        "init_coords": copy.deepcopy(all_coords4input),
        "init_abc": copy.deepcopy(all_opt_abc),
        "init_angles": copy.deepcopy(all_angles),
        "atomic_mask": None,
        "radii_tensor": None,
        "atom_mask_for_all_ox_states": atom_mask_for_all_ox_states,
        "radii_for_all_ox_states": radii_for_all_ox_states,
        "ox_states_used_mask": ox_states_used_mask,
        "site_ids": site_ids,
        "fnames": fnames,
    }

    if use_atomic_mask:
        crystalformer_all_inputs_dict["atomic_mask"] = atom_mask_for_all_ox_states.to(
            device
        )
        ox_mask_learnable_tensor_per_crystal = create_learnable_oxides_mask(
            crystalformer_all_inputs_dict
        )
        crystalformer_all_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
            ox_mask_learnable_tensor_per_crystal
        )
        crystalformer_all_inputs_dict["radii_tensor"] = radii_for_all_ox_states.to(
            device
        )

    # divide the initial data into small batches
    minibatch_datas = []
    for min_id in range(0, num_candidate, batch_size):
        # select batch candidate crystals
        mini_batch_inputs_dict = select_batch_candidates_crystalformer(
            min_id=min_id,
            num_batch_crystal=batch_size,
            crystalformer_all_inputs_dict=crystalformer_all_inputs_dict,
        )
        scalers = {
            "abc_scaler": ABC_Scaler(
                init_batch_abc=mini_batch_inputs_dict["batch_abc"],
                min_length=abc_range[0],
                max_length=abc_range[1],
                device=device,
            ),
            "angle_scaler": Angle_Scaler(
                min_angle=angle_range[0],
                max_angle=angle_range[1],
            ),
        }
        minibatch_datas.append([mini_batch_inputs_dict, scalers])

    return minibatch_datas


# In[ ]:


def create_crystalformer_init_structure_from_directory(
    dir_path: str,
    intermid_dir_path: str,
    max_ox_states: int,
    mask_method: str,
    crystal_system: Optional[str],
    num_candidate: Optional[int] = None,
):
    """Read the packaged structures and frozen masks in their recorded order."""
    if crystal_system is not None:
        raise ValueError("Section 2.1 requires the non-perovskite input dataset")
    input_dir = Path(dir_path)
    file_names = (input_dir / "template_order.txt").read_text().splitlines()
    if not file_names or len(set(file_names)) != len(file_names):
        raise ValueError("template_order.txt must contain unique structure filenames")
    if any(
        Path(name).name != name or not name.endswith(".vasp") for name in file_names
    ):
        raise ValueError("template_order.txt must contain local .vasp filenames")
    if num_candidate is None:
        num_candidate = len(file_names)
    if not 1 <= num_candidate <= len(file_names):
        raise ValueError(f"num_candidate must be between 1 and {len(file_names)}")

    with np.load(
        Path(intermid_dir_path) / "initial_masks.npz", allow_pickle=False
    ) as data:
        if data["fnames"].tolist() != file_names:
            raise ValueError("Frozen mask filenames do not match template_order.txt")
        sizes = data["sizes"].copy()
        if sizes.shape != (len(file_names),) or np.any(sizes <= 0):
            raise ValueError("Frozen masks contain invalid structure sizes")
        total_atoms = int(sizes.sum())
        expected_shapes = {
            "atomic_mask": (total_atoms, 98, max_ox_states),
            "radii_tensor": (total_atoms, 98, max_ox_states),
            "ox_states_used_mask": (total_atoms, max_ox_states),
            "site_ids": (total_atoms,),
        }
        for key, shape in expected_shapes.items():
            if data[key].shape != shape:
                raise ValueError(
                    f"Invalid frozen mask shape for {key}: {data[key].shape}"
                )
        selected_atoms = int(sizes[:num_candidate].sum())
        masks = {
            key: torch.tensor(data[key][:selected_atoms], dtype=torch.float32)
            for key in expected_shapes
        }

    fnames = file_names[:num_candidate]
    structures = [Atoms.from_poscar(str(input_dir / name)) for name in fnames]
    actual_sizes = [len(atoms.elements) for atoms in structures]
    if actual_sizes != sizes[:num_candidate].tolist():
        raise ValueError("Structure atom counts do not match the frozen masks")
    size = torch.tensor(actual_sizes, dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(num_candidate), size)
    abc = torch.tensor([atoms.lattice.abc for atoms in structures], dtype=torch.float32)
    angles = torch.tensor(
        [atoms.lattice.angles for atoms in structures], dtype=torch.float32
    )
    coords = torch.tensor(
        np.concatenate([atoms.coords for atoms in structures]), dtype=torch.float32
    )
    atom_dist = torch.full((selected_atoms, 98), 1.0 / 98)
    return (
        abc,
        atom_dist,
        batch,
        coords,
        size,
        angles,
        fnames,
        masks["atomic_mask"],
        masks["radii_tensor"],
        masks["ox_states_used_mask"],
        masks["site_ids"],
    )


def pickup_candidates_crystalformer(
    min_id: int,
    max_id: int,
    crystalformer_all_inputs_dict: dict,
):
    assert min_id < max_id

    x = crystalformer_all_inputs_dict["x"]
    batch = crystalformer_all_inputs_dict["batch"]
    coords4input = crystalformer_all_inputs_dict["coords4input"]
    size = crystalformer_all_inputs_dict["size"]
    abc = crystalformer_all_inputs_dict["abc"]
    angles = crystalformer_all_inputs_dict["angles"]
    atomic_mask = crystalformer_all_inputs_dict["atomic_mask"]
    radii_tensor = crystalformer_all_inputs_dict["radii_tensor"]
    fnames = np.array(crystalformer_all_inputs_dict["fnames"])
    ox_mask_learnable_tensor_per_crystal = crystalformer_all_inputs_dict[
        "ox_mask_learnable_tensor_per_crystal"
    ]
    ox_states_used_mask = crystalformer_all_inputs_dict["ox_states_used_mask"]
    site_ids = crystalformer_all_inputs_dict["site_ids"]

    selected_atom_ids = (min_id <= batch.detach().cpu().numpy()) & (
        batch.detach().cpu().numpy() < max_id
    )
    selected_x = x[selected_atom_ids]
    selected_batch = batch[selected_atom_ids]
    selected_coords4input = coords4input[selected_atom_ids]
    selected_site_ids = site_ids[selected_atom_ids]

    unique_ids = np.unique(batch.detach().cpu().numpy())
    selected_ids = (min_id <= unique_ids) & (unique_ids < max_id)
    selected_abc = abc[selected_ids]
    selected_angles = angles[selected_ids]
    selected_size = size[selected_ids]
    selected_fnames = fnames[selected_ids]

    if atomic_mask is not None:
        selected_atomic_mask = atomic_mask[selected_atom_ids]
        selected_ox_mask_learnable_tensor_per_crystal = (
            ox_mask_learnable_tensor_per_crystal[selected_ids]
        )
        selected_ox_states_used_mask = ox_states_used_mask[selected_atom_ids]
    else:
        selected_atomic_mask = None
        selected_ox_mask_learnable_tensor_per_crystal = None
        selected_ox_states_used_mask = None

    if radii_tensor is not None:
        selected_radii_tensor = radii_tensor[selected_atom_ids]
    else:
        selected_radii_tensor = None

    minibatch_inputs_dict = {
        "atomic_distribution": selected_x,
        "batch": selected_batch,
        "batch_dir_coords": selected_coords4input,
        "size": selected_size,
        "batch_abc": selected_abc,
        "batch_angle": selected_angles,
        "init_coords": copy.deepcopy(selected_coords4input),
        "init_abc": copy.deepcopy(selected_abc),
        "init_angles": copy.deepcopy(selected_angles),
        "fnames": selected_fnames,
        "atomic_mask": selected_atomic_mask,
        "radii_tensor": selected_radii_tensor,
        "ox_mask_learnable_tensor_per_crystal": selected_ox_mask_learnable_tensor_per_crystal,
        "ox_states_used_mask": selected_ox_states_used_mask,
        "site_ids": selected_site_ids,
    }

    return minibatch_inputs_dict


# In[ ]:


def select_batch_candidates_crystalformer(
    min_id: int,
    num_batch_crystal: int,
    crystalformer_all_inputs_dict: dict,
):
    """
    最適化対象を小さな塊に分割する
    Args:
        min_id: 選別する最適化対象の最小のID
        num_batch_crystal: 一度の最適化で最適化する結晶の数
        crystalformer_all_inputs_dict: Crystalformerへの入力
    Returns:
        opt_x: 最適化対象の原子種の分布. shape = (size, num_atom_type)
        batch: 最適化対象の結晶のID. shape = (num_batch_crystal,)
        coords4input: 最適化対象の原子の格子内座標. shape =(num_batch_crystal内の総原子数==size, 3)
        size: 最適化対象の結晶に含まれる原子数. shape = (num_batch_crystal,)
        opt_abc: 最適化対象の結晶の格子ベクトルの長さ. shape = (size, 3)
        angles: 最適化対象の結晶の格子ベクトルの角度. shape = (sizel, 3)
    """

    max_id = min_id + num_batch_crystal
    mini_batch_inputs_dict = pickup_candidates_crystalformer(
        min_id=min_id,
        max_id=max_id,
        crystalformer_all_inputs_dict=crystalformer_all_inputs_dict,
    )
    return mini_batch_inputs_dict


# ### データ読み込みの統合

# In[ ]:


def check_alignment_between_alignn_dataset_and_crystalformer_dataset(
    alignn_data,
    crystalformer_data,
):
    for key in set(alignn_data[0][0].keys()) & set(crystalformer_data[0][0].keys()):
        if key == "fnames":
            assert (
                alignn_data[0][0]["fnames"].astype(str)
                == crystalformer_data[0][0]["fnames"].astype(str)
            ).all()
        else:
            torch.testing.assert_close(
                torch.tensor(alignn_data[0][0][key]).cpu(),
                torch.tensor(crystalformer_data[0][0][key]).cpu(),
                atol=3e-5,
                rtol=5e-6,
            )

    torch.testing.assert_close(
        alignn_data[0][1]["abc_scaler"].scale(alignn_data[0][0]["batch_abc"].cuda()),
        crystalformer_data[0][1]["abc_scaler"].scale(
            crystalformer_data[0][0]["batch_abc"].cuda()
        ),
    )
    torch.testing.assert_close(
        alignn_data[0][1]["angle_scaler"].scale(alignn_data[0][0]["batch_angle"].cpu()),
        crystalformer_data[0][1]["angle_scaler"].scale(
            crystalformer_data[0][0]["batch_angle"].cpu()
        ),
    )


# In[ ]:


def load_initial_data(
    settings_dict: Dict[str, Any],
    model_name: str,
    dir_path: str,
    intermid_dir_path: str,
    use_atomic_mask: bool,
    max_ox_states: int,
    mask_method: str,
    test_mode: bool,
    crystal_system: Optional[str],
    num_candidate: Optional[int],
    batch_size: int,
    angle_range: Tuple[float, float],
    abc_range: Tuple[float, float],
    graph_update_mode: bool = False,
    device: str = "cuda",
):
    if model_name == "Crystalformer":
        # load all initial data
        minibatch_datas = load_crystalformer_initial_data(
            device=device,
            settings_dict=settings_dict,
            dir_path=dir_path,
            intermid_dir_path=intermid_dir_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            crystal_system=crystal_system,
            mask_method=mask_method,
            angle_range=angle_range,
            abc_range=abc_range,
            test_mode=test_mode,
        )
    elif model_name == "ALIGNN":
        minibatch_datas = load_ALIGNN_initial_data(
            settings_dict=settings_dict,
            dir_path=dir_path,
            intermid_dir_path=intermid_dir_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            crystal_system=crystal_system,
            mask_method=mask_method,
            angle_range=angle_range,
            abc_range=abc_range,
            graph_update_mode=graph_update_mode,
            test_mode=test_mode,
        )
    elif model_name == "Both":
        minibatch_datas_crystalformer = load_crystalformer_initial_data(
            dir_path=dir_path,
            intermid_dir_path=intermid_dir_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            crystal_system=crystal_system,
            mask_method=mask_method,
            angle_range=angle_range,
            abc_range=abc_range,
            test_mode=test_mode,
        )
        minibatch_datas_alignn = load_ALIGNN_initial_data(
            dir_path=dir_path,
            intermid_dir_path=intermid_dir_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            crystal_system=crystal_system,
            mask_method=mask_method,
            angle_range=angle_range,
            abc_range=abc_range,
            graph_update_mode=graph_update_mode,
            test_mode=test_mode,
        )
        assert len(minibatch_datas_crystalformer) == len(minibatch_datas_alignn)
        check_alignment_between_alignn_dataset_and_crystalformer_dataset(
            minibatch_datas_alignn, minibatch_datas_crystalformer
        )

        minibatch_datas = []
        for mb_crystalformer, mb_alignn in zip(
            minibatch_datas_crystalformer, minibatch_datas_alignn
        ):
            minibatch_dict = mb_crystalformer[0] | mb_alignn[0]
            scalers = mb_crystalformer[1]
            minibatch_datas.append([minibatch_dict, scalers])
    else:
        raise NotImplementedError(f"model_name, {model_name}, is not implemented yet.")

    return minibatch_datas


# In[ ]:


# In[ ]:


# ## モデルの読み込み

# ### ALIGNNの読み込み

# In[ ]:


"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
# from dgl.nn.functional import edge_softmax
# from pydantic.typing import Literal
from typing import Literal, Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from torch import nn
from torch.nn import functional as F

from third_party.alignn_minimal.models.utils import RBFExpansion
from pydantic_settings import BaseSettings


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    num_classes: int = 2
    extra_features: int = 0

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN4inverse(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.config = config
        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(config.hidden_features, config.hidden_features)
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()
        self.readout_feat = AvgPooling()
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

        if config.extra_features != 0:
            # Credit for extra_features work:
            # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
            self.extra_feature_embedding = MLPLayer(
                config.extra_features, config.extra_features
            )
            self.fc3 = nn.Linear(
                config.hidden_features + config.extra_features,
                config.output_features,
            )
            self.fc1 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )
            self.fc2 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self,
        g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
        atom_features,
        bondlength,
        angle_features,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            # print('features2',features.shape)

            g, lg = g
            lg = lg.local_var()
            """Commented out and changed for inverse problem
            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))
            """
            z = self.angle_embedding(angle_features)

        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            # print('g',g)
            # print('features1',features.shape)
            features = self.extra_feature_embedding(features)

        """ Commented out and changed for inverse problem
        g = g.local_var()
        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        # print('x1',x.shape)
        x = self.atom_embedding(x)
        # print('x2',x.shape)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)"""
        x = self.atom_embedding(atom_features)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        # print('h',h.shape)
        # print('features',features.shape)
        if self.config.extra_features != 0:
            h_feat = self.readout_feat(g, features)
            # print('h1',h.shape)
            # print('h_feat',h_feat.shape)
            h = torch.cat((h, h_feat), 1)
            # print('h2',h.shape)

            h = self.fc1(h)

            h = self.fc2(h)

            out = self.fc3(h)
        else:
            out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)


# In[ ]:


# from alignn.data import get_torch_dataset
import json
import os
import tempfile
import zipfile

import requests
import torch
from jarvis.core.atoms import Atoms

from third_party.alignn_minimal.graphs import Graph
from third_party.alignn_minimal.models.alignn import ALIGNN, ALIGNNConfig


def figshare_model_list():
    return {
        "mp_e_form_alignnn": [
            "https://figshare.com/ndownloader/files/31458811",
            1,
        ],
        "mp_gappbe_alignnn": [
            "https://figshare.com/ndownloader/files/31458814",
            1,
        ],
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


# In[ ]:


def Load_Pretrained_ALIGNN():
    bandgap_model = get_figshare_model_for_inverse_problem(
        model_name="mp_gappbe_alignnn"
    )
    e_form_model = get_figshare_model_for_inverse_problem(
        model_name="mp_e_form_alignnn"
    )
    return bandgap_model, e_form_model


# In[ ]:


def atom_dist_to_features(atomic_distribution, atom_feat_matrix):
    """
    Convert atomic distribution to atomic features.
    """
    atomic_features = torch.matmul(atomic_distribution, atom_feat_matrix)
    return atomic_features


# ### Crystalformerの読み込み

# In[ ]:


from crystalformer.models.latticeformer import Latticeformer
from crystalformer.utils import Params


def load_pretrained_crystalformer(
    task: str, layer: int, weight_path_dir: str, device: str = "cuda"
):
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
        ckeckpoint = torch.load(f, weights_only=False, map_location="cpu")
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


# In[ ]:


def Load_Pretrained_Crystalformers(device: str = "cuda"):
    # モデルの読み込み
    bandgap_model = load_pretrained_crystalformer(
        task="bandgap",  # 'e_form' or 'bandgap'
        layer=4,
        device=device,
        weight_path_dir=os.path.join(get_master_dir(), f"models/crystalformer"),
    ).to(device)
    e_form_model = load_pretrained_crystalformer(
        task="e_form",  # 'e_form' or 'bandgap'
        layer=4,
        device=device,
        weight_path_dir=os.path.join(get_master_dir(), f"models/crystalformer"),
    ).to(device)
    return bandgap_model, e_form_model


# In[ ]:


def Load_Pretrained_Both():
    bandgap_model_alignn, e_form_model_alignn = Load_Pretrained_ALIGNN()
    bandgap_model_crystalformer, e_form_model_crystalformer = (
        Load_Pretrained_Crystalformers()
    )
    bandgap_models = {
        "ALIGNN": bandgap_model_alignn,
        "Crystalformer": bandgap_model_crystalformer,
    }
    e_form_models = {
        "ALIGNN": e_form_model_alignn,
        "Crystalformer": e_form_model_crystalformer,
    }
    return bandgap_models, e_form_models


# In[ ]:


from jarvis.core.specie import get_node_attributes
from torch.nn import Parameter

# ## 原子種の定義や結晶ベクトルの変換関数を定義

# In[ ]:


def create_element_list():
    VALID_ELEMENTS98 = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
    ][:98]
    return VALID_ELEMENTS98


# In[ ]:


def direct_to_cartesian(
    dir_coords: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert direct coordinates to cartesian coordinates using lattice vectors.

    The conversion is done by matrix multiplying the direct coordinates with the lattice vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates, with shape (N, 3), where N is the number
                                   of atoms in the structure.
        lattice_vectors (torch.Tensor): The lattice vectors, with shape (3, 3).

    Returns:
        torch.Tensor: The cartesian coordinates, with the same shape as dir_coords.
    """
    return torch.matmul(dir_coords, lattice_vectors)


# In[ ]:


def compute_abc_angle(batch_lattice_vec: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Computes the side lengths and angles of crystal lattices from their lattice vectors for a batch.

    This function calculates the side lengths (a, b, c) and angles (alpha, beta, gamma) of the crystal lattices
    based on the provided lattice vectors for each item in the batch. The angles are computed in degrees.

    Args:
        batch_lattice_vec (torch.Tensor): A tensor containing the lattice vectors for the crystal lattices in the batch.
                                          The shape is (B, 3, 3), where B is the batch size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                           - The first tensor contains the side lengths (a, b, c) of the crystal lattices in the batch, with shape (B, 3).
                                           - The second tensor contains the angles (alpha, beta, gamma) in degrees of the crystal lattices in the batch, with shape (B, 3).
    """
    B = batch_lattice_vec.shape[0]
    batch_abc = torch.zeros(
        (B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device
    )
    batch_angle = torch.zeros(
        (B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device
    )

    # 辺の長さを計算
    batch_abc[:, 0] = torch.norm(batch_lattice_vec[:, 0, :], dim=1)
    batch_abc[:, 1] = torch.norm(batch_lattice_vec[:, 1, :], dim=1)
    batch_abc[:, 2] = torch.norm(batch_lattice_vec[:, 2, :], dim=1)

    # 成す角を計算
    # alpha = angle between b and c
    batch_angle[:, 0] = torch.acos(
        torch.sum(batch_lattice_vec[:, 1, :] * batch_lattice_vec[:, 2, :], dim=1)
        / (batch_abc[:, 1] * batch_abc[:, 2])
    )
    # beta = angle between a and c
    batch_angle[:, 1] = torch.acos(
        torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 2, :], dim=1)
        / (batch_abc[:, 0] * batch_abc[:, 2])
    )
    # gamma = angle between a and b
    batch_angle[:, 2] = torch.acos(
        torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 1, :], dim=1)
        / (batch_abc[:, 0] * batch_abc[:, 1])
    )

    # ラジアンから度へ変換
    batch_angle = torch.rad2deg(batch_angle)

    return batch_abc, batch_angle


# In[ ]:


# ## 最適化に必要な関数など

# ### CrystalformerとALIGNN共用

# In[ ]:


def temperature_softmax(logits, temperature=1.0, dim=-1):
    """Applies a temperature-scaled softmax to the input logits.

    This function modifies the softmax operation by scaling the logits with
    a temperature parameter before applying softmax. The temperature parameter
    can adjust the sharpness of the output distribution. A higher temperature
    makes the distribution more uniform, while a lower temperature makes it
    sharper.

    Args:
        logits (torch.Tensor): The input logits to which softmax will be applied.
        temperature (float, optional): The temperature to scale the logits. Default is 1.0.

    Returns:
        torch.Tensor: The softmax output after applying temperature scaling.

    Raises:
        ValueError: If the temperature is non-positive.

    Example:
        >>> logits = torch.tensor([2.0, 1.0, 0.1])
        >>> temperature = 0.5
        >>> softmax_outputs = temperature_softmax(logits, temperature)
        >>> print(softmax_outputs)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    # Adjust logits based on the temperature
    adjusted_logits = logits / temperature
    # Apply softmax to the adjusted logits
    return torch.softmax(adjusted_logits, dim=dim)


# In[ ]:


from typing import Dict, List, Optional, Tuple


class GapLoss(torch.nn.Module):
    def __init__(
        self,
        gap_min: Optional[float] = None,
        gap_max: Optional[float] = None,
        mode: str = "band",
    ):
        super().__init__()
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.mode = mode

        if gap_min is None and gap_max is None:
            raise ValueError("gap_min and gap_max cannot be both None")

        if mode not in ["band", "min", "max"]:
            raise ValueError("mode must be 'band', 'min' or 'max'")

        if mode == "band" and gap_min is not None and gap_max is not None:
            self.mean = (gap_min + gap_max) / 2
            self.margin = (gap_max - gap_min) / 2
        elif mode == "min" and type(gap_min) == float and gap_max is None:
            pass
        elif mode == "max" and type(gap_max) == float and gap_min is None:
            pass
        else:
            raise ValueError(
                f"mode and gap_min, gap_max are not matched! mode: {mode}, gap_min: {gap_min}, gap_max: {gap_max}"
            )

    def forward(self, x: torch.Tensor, reduction="mean") -> torch.Tensor:
        if self.mode == "band":
            loss = self.band_loss(x)
        elif self.mode == "min":
            loss = self.min_loss(x)
        elif self.mode == "max":
            loss = self.max_loss(x)
        else:
            raise ValueError("mode must be 'band', 'min' or 'max'")

        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "sum":
            return torch.sum(loss)
        elif reduction == "none":
            return loss
        else:
            raise ValueError("reduction must be 'mean', 'sum' or 'none")

    def min_loss(self, x):
        return torch.clip(x - self.gap_min, min=0).squeeze()

    def max_loss(self, x):
        return torch.clip(self.gap_max - x, min=0).squeeze()

    def band_loss(self, x):
        return torch.clip(torch.abs(x - self.mean) - self.margin, min=0).squeeze()


# In[ ]:


def set_schedule(
    num_steps: int,
    ef_coef_base: float = 1.0,
    T_min: float = 0.0001,
    T_max: float = 0.01,
):
    # 原子のソフトマックス温度など

    # E_formの損失を追加＋softmax温度のスケジュール化
    dist_temp_sche = [
        T_min + (T_max - T_min) / num_steps * i for i in range(num_steps + 1)
    ][::-1]
    Ef_coef_schedule = [ef_coef_base] * num_steps
    step_of_x_opt_stop = 1e8

    return dist_temp_sche, Ef_coef_schedule, step_of_x_opt_stop


# In[ ]:


class Angle_Scaler:
    def __init__(self, min_angle: float, max_angle: float):
        self.min_angle = min_angle  #'degree'
        self.max_angle = max_angle  #'degree'

    def scale(self, angles):
        return (angles - self.min_angle) / (self.max_angle - self.min_angle)

    def rescale(self, scaled_batch_angle):
        return (
            torch.clip(scaled_batch_angle, min=0.0, max=1.0)
            * (self.max_angle - self.min_angle)
            + self.min_angle
        )


class ABC_Scaler:
    def __init__(
        self, init_batch_abc, min_length: float, max_length: float, device="cuda"
    ):
        self.init_batch_abc = init_batch_abc
        self.base_length = (
            (torch.max(self.init_batch_abc, dim=1).values).view(-1, 1).to(device)
        )
        self.min_length = min_length
        self.max_length = max_length
        self.device = device
        ## あとで消す
        # self.base_length = self.max_length

    def update_base_length(self):
        self.base_length = (
            (torch.max(self.init_batch_abc, dim=1).values).view(-1, 1).to(self.device)
        )
        ## あとで消す
        # self.base_length = self.max_length

    def scale(self, batch_abc):
        return batch_abc / self.base_length

    def rescale(self, scaled_batch_abc):
        return torch.clip(
            scaled_batch_abc * self.base_length,
            min=self.min_length,
            max=self.max_length,
        )


# In[ ]:


import torch
import torch.nn as nn


class toleranceLoss(nn.Module):
    def __init__(self, low=0.8, high=1.0, margin=0.1):
        super(toleranceLoss, self).__init__()
        self.low = low + margin
        self.high = high - margin

    def forward(self, predictions):
        """
        Compute custom loss.

        Args:
            predictions (torch.Tensor): Tensor containing the predicted values.

        Returns:
            torch.Tensor: Tensor containing the computed loss.
        """
        # Ensure predictions are within a valid range by applying thresholds
        lower_mask = predictions < self.low
        upper_mask = predictions > self.high

        # Calculate differences from the nearest boundary
        loss = torch.where(lower_mask, self.low - predictions, predictions - self.high)

        # Apply mask to zero out losses within the acceptable range
        loss = torch.where(
            (predictions >= self.low) & (predictions <= self.high),
            torch.zeros_like(predictions),
            loss,
        )

        return loss


# Example usage


# ### Crystalformerのステップごとの前処理関数

# In[ ]:


def calculate_cartesian_vec(
    num_candidate: int,
    size: torch.Tensor,
    coords4input: torch.Tensor,
    abc: torch.Tensor,
    angle: torch.Tensor,
):
    """
    デカルト座標の格子内座標(pos)と格子ベクトル(trans_vec)を計算する
    Args:
        num_candidate: 最適化する解候補(結晶)の数 == N
        size: 各結晶に含まれる原子数. shape = (N,)
        coords4input: 各結晶の原子の格子内座標. shape =(解候補内の総原子数==size, 3)
        abc: 各結晶の格子ベクトルの長さ. shape = (N, 3)
        angle: 各結晶の格子ベクトルの角度. shape = (N, 3)
    Returns:
        pos: デカルト座標の格子内座標. shape = (解候補内の総原子数==size, 3)
        trans_vec: デカルト座標の格子ベクトル. shape = (N, 3, 3)
    """
    # (stepごとの処理) 更新された結晶格子から格子ベクトルの計算をし、原子座標を直交座標に変換
    trans_vec = compute_lattice_vectors(abc, angle)

    # 内部座標を0~1の値にclipする
    coords4input = torch.clip(coords4input, 0, 1)

    # trans_vecを結晶数だけ展開し、各原子座標を直交座標に変換
    trans_vec4input = torch.cat(
        [
            trans_vec[i, ...].unsqueeze(0).repeat(size[i], 1, 1)
            for i in range(num_candidate)
        ],
        dim=0,
    )
    pos = direct_to_cartesian_batch(
        dir_coords=coords4input, lattice_vectors=trans_vec4input
    )
    return pos, trans_vec


# In[ ]:


def Crystalformer_preprocess(
    scaled_batch_abc: torch.Tensor,
    scaled_batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    atomic_distribution: torch.Tensor,
    init_coords: torch.Tensor,
    size: torch.Tensor,
    temperature: float,
    abc_scaler: ABC_Scaler,
    angle_scaler: Angle_Scaler,
    atomic_mask: Optional[torch.Tensor],
    ox_states_used_mask: Optional[torch.Tensor],
    limit_coords_displacement: Optional[float],
    ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor],
    adding_noise_scale: Optional[float],
    abc_range: Tuple[float, float],
    angle_range: Tuple[float, float],
):
    # add noise
    if adding_noise_scale is not None:
        noise_dir_coords = (
            torch.rand_like(batch_dir_coords, device=batch_dir_coords.device) * 2 - 1.0
        ) * adding_noise_scale
        noise_scaled_abc = (
            torch.rand_like(scaled_batch_abc, device=scaled_batch_abc.device) * 2 - 1.0
        ) * adding_noise_scale
        noise_scaled_angle = (
            torch.rand_like(scaled_batch_angle, device=scaled_batch_angle.device) * 2
            - 1.0
        ) * adding_noise_scale
    else:
        noise_dir_coords = 0
        noise_scaled_abc = 0
        noise_scaled_angle = 0

    # stepごとの前処理
    batch_abc = abc_scaler.rescale(scaled_batch_abc + noise_scaled_abc)
    batch_angle = angle_scaler.rescale(scaled_batch_angle + noise_scaled_angle)
    ## 制限をかける
    batch_abc = torch.clamp(batch_abc, min=abc_range[0], max=abc_range[1])
    batch_angle = torch.clamp(batch_angle, min=angle_range[0], max=angle_range[1])

    # 初期値(init_coords)との差分をlimit_coords_displacementに留め, 内部座標の値を周期的境界条件で0~1に制限する
    if limit_coords_displacement is not None:
        normed_batch_dir_coords = torch.clamp(
            batch_dir_coords + noise_dir_coords,
            min=init_coords - limit_coords_displacement,
            max=init_coords + limit_coords_displacement,
        )
    else:
        normed_batch_dir_coords = batch_dir_coords
    normed_batch_dir_coords = torch.remainder(
        normed_batch_dir_coords + noise_dir_coords, 1.0
    )  # 内部座標の値を周期的境界条件で0~1に制限する

    if ox_mask_learnable_tensor_per_crystal is not None:
        stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
            ox_mask_learnable_tensor_per_crystal, size
        )
        normalized_dist, sharpened_ox_mask = masking_atomic_distribution(
            atomic_distribution,
            atomic_mask,
            ox_states_used_mask,
            stacked_learnable_ox_weight,
            temperature,
        )
    else:
        sharpened_ox_mask = torch.zeros_like(ox_states_used_mask) * torch.nan
        normalized_dist = temperature_softmax(
            atomic_distribution, temperature=temperature
        )

    pos, trans_vec = calculate_cartesian_vec(
        num_candidate=size.shape[0],
        size=size,
        coords4input=normed_batch_dir_coords,
        abc=batch_abc,
        angle=batch_angle,
    )
    return (
        batch_abc,
        batch_angle,
        normed_batch_dir_coords,
        normalized_dist,
        pos,
        trans_vec,
        sharpened_ox_mask,
    )


# ### ALIGNNのステップごとの前処理関数

# In[ ]:


def masking_atomic_distribution(
    atomic_distribution: torch.Tensor,
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    stacked_learnable_ox_weight: torch.Tensor,
    softmax_temp: float,
    eps: float = 1e-6,
):
    """
    Mask atomic distribution with atomic mask.
    Args:
        atomic_distribution: atomic distribution. shape = (N, num_type_of_atoms)
        atomic_mask: atomic mask. shape = (N, num_type_of_atoms) or (N, num_type_of_atoms, max_ox_state)
        ox_states_used_mask: mask for used oxidation states. a learnable parameter. shape = (N, max_ox_state)
        stacked_learnable_ox_weight: learnable weight for oxidation states. shape = (N, max_ox_state)
        softmax_temp: temperature for softmax

    """
    assert (
        atomic_distribution.shape[0]
        == atomic_mask.shape[0]
        == ox_states_used_mask.shape[0]
        == stacked_learnable_ox_weight.shape[0]
    )
    assert atomic_distribution.shape[1] == atomic_mask.shape[1]
    assert (
        ox_states_used_mask.shape[1]
        == atomic_mask.shape[2]
        == stacked_learnable_ox_weight.shape[1]
    )
    # multiple masks for each atomic feature
    # atomic_distribution: (N, num_type_of_atoms)
    # atomic_mask: (N, num_type_of_atoms, max_ox_state)

    atomic_distribution = torch.clip(
        atomic_distribution, min=eps
    )  # (N, num_type_of_atoms), clip to avoid minus value for selected atoms
    stacked_learnable_ox_weight = torch.clip(
        stacked_learnable_ox_weight, min=eps
    )  # (N, max_ox_state), clip to avoid minus value for selected oxidation states

    masked_atomic_distribution = (
        atomic_distribution.unsqueeze(-1) * atomic_mask
    )  # (N, num_type_of_atoms, max_ox_state)
    masked_atomic_distribution = temperature_softmax(
        masked_atomic_distribution, temperature=softmax_temp, dim=1
    )  # (N, num_type_of_atoms, max_ox_state)
    sharpened_ox_mask = temperature_softmax(
        ox_states_used_mask * stacked_learnable_ox_weight,
        temperature=softmax_temp,
        dim=1,
    ).unsqueeze(
        1
    )  # (N, 1, max_ox_state)
    masked_atomic_distribution = torch.sum(
        masked_atomic_distribution * sharpened_ox_mask, dim=-1
    )  # (N, num_type_of_atoms)
    masked_atomic_distribution = temperature_softmax(
        masked_atomic_distribution, temperature=softmax_temp, dim=1
    )  # (N, num_type_of_atoms)

    return masked_atomic_distribution, sharpened_ox_mask.squeeze(1)


# In[ ]:


def ALIGNN_preprocess(
    scaled_batch_abc: torch.Tensor,
    scaled_batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    atomic_distribution: torch.Tensor,
    batch_dst_ids: torch.Tensor,
    batch_src_ids: torch.Tensor,
    batch_displace: torch.Tensor,
    z_src_ids: torch.Tensor,
    z_dst_ids: torch.Tensor,
    size: torch.Tensor,
    init_coords: torch.Tensor,
    temperature: float,
    abc_scaler: ABC_Scaler,
    angle_scaler: Angle_Scaler,
    num_edges: int,
    atom_feat_matrix: torch.Tensor,
    limit_coords_displacement: Optional[float],
    adding_noise_scale: Optional[float],
    atomic_mask: Optional[torch.Tensor] = None,
    ox_states_used_mask: Optional[torch.Tensor] = None,
    ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor] = None,
    value_round: Optional[int] = None,
):
    """
    ALIGNNに入れる前の処理を行う関数
    """
    if adding_noise_scale is not None:
        noise_dir_coords = (
            torch.rand_like(batch_dir_coords, device=batch_dir_coords.device) * 2 - 1.0
        ) * adding_noise_scale
        noise_scaled_abc = (
            torch.rand_like(scaled_batch_abc, device=scaled_batch_abc.device) * 2 - 1.0
        ) * adding_noise_scale
        noise_scaled_angle = (
            torch.rand_like(scaled_batch_angle, device=scaled_batch_angle.device) * 2
            - 1.0
        ) * adding_noise_scale
    else:
        noise_dir_coords = 0
        noise_scaled_abc = 0
        noise_scaled_angle = 0

    batch_abc = abc_scaler.rescale(scaled_batch_abc + noise_scaled_abc)
    batch_angle = angle_scaler.rescale(scaled_batch_angle + noise_scaled_angle)

    # 初期値(init_coords)との差分をlimit_coords_displacementに留め, 内部座標の値を周期的境界条件で0~1に制限する
    if limit_coords_displacement is not None:
        normed_batch_dir_coords = torch.clamp(
            batch_dir_coords + noise_dir_coords,
            min=init_coords - limit_coords_displacement,
            max=init_coords + limit_coords_displacement,
        )
    else:
        normed_batch_dir_coords = batch_dir_coords
    normed_batch_dir_coords = torch.remainder(
        normed_batch_dir_coords + noise_dir_coords, 1.0
    )  # 内部座標の値を周期的境界条件で0~1に制限する

    if ox_mask_learnable_tensor_per_crystal is not None:
        stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
            ox_mask_learnable_tensor_per_crystal, size
        )
        normalized_dist, sharpened_ox_mask = masking_atomic_distribution(
            atomic_distribution,
            atomic_mask,
            ox_states_used_mask,
            stacked_learnable_ox_weight,
            temperature,
        )
    else:
        sharpened_ox_mask = torch.zeros_like(ox_states_used_mask) * torch.nan
        normalized_dist = temperature_softmax(
            atomic_distribution, temperature=temperature
        )

    atomic_features = atom_dist_to_features(normalized_dist, atom_feat_matrix)
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle)
    batch_lattice_vectors = create_batch_lattice_vectors(lattice_vectors, num_edges)
    if value_round is not None:
        batch_lattice_vectors = torch.round(batch_lattice_vectors, decimals=value_round)
        normed_batch_dir_coords = torch.round(
            normed_batch_dir_coords, decimals=value_round
        )

    edge_vectors, bondlength = compute_bondlength(
        normed_batch_dir_coords,
        batch_dst_ids,
        batch_src_ids,
        batch_displace,
        batch_lattice_vectors,
    )
    bond_angles = calculate_bond_cosine(
        -edge_vectors[z_src_ids], edge_vectors[z_dst_ids]
    )

    if value_round is not None:
        bond_angles = torch.round(bond_angles, decimals=value_round)
        bondlength = torch.round(bondlength, decimals=value_round)

    return (
        atomic_features,
        bondlength,
        bond_angles,
        batch_abc,
        batch_angle,
        lattice_vectors,
        normalized_dist,
        normed_batch_dir_coords,
        sharpened_ox_mask,
    )


# ## 全体の初期化と最適化

# ### ALIGNN, Crystalformer共用

# In[ ]:


import copy


def noise_robust_test(
    optimized_mini_batch_inputs_dict: Dict[str, torch.Tensor],
    noise_scale: float,
    evaluation_for_each_batch,
    scalers,
    bandgap_model,
    e_form_model,
    Ef_coef_schedule,
    gap_loss_func,
    crystal_system,
    tolerance_loss_func,
    e_form_min,
):
    noise_robust_dict = copy.deepcopy(optimized_mini_batch_inputs_dict)
    num_test = 20

    if "bandgap_onehot" in noise_robust_dict.keys():
        # single model
        noised_predictions_band, noised_predictions_e_form = [], []
        model_type = "single"
    else:
        # two models
        noised_predictions_band_alignn, noised_predictions_e_form_alignn = [], []
        (
            noised_predictions_band_crystalformer,
            noised_predictions_e_form_crystalformer,
        ) = (
            [],
            [],
        )
        model_type = "two"

    for _ in tqdm(
        range(num_test),
        desc=f"noise_robust_test. noise scale:{noise_scale}",
        leave=False,
    ):
        noise_robust_dict = evaluation_for_each_batch(
            optimized_mini_batch_inputs_dict=noise_robust_dict,
            scalers=scalers,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            gap_loss_func=gap_loss_func,
            crystal_system=crystal_system,
            tolerance_loss_func=tolerance_loss_func,
            adding_noise_scale=noise_scale,
            e_form_min=e_form_min,
            Ef_coef_schedule=Ef_coef_schedule,
            limit_coords_displacement=None,
        )

        if model_type == "single":
            # single model
            noised_predictions_band.append(noise_robust_dict["bandgap_onehot"])
            noised_predictions_e_form.append(noise_robust_dict["ef_onehot"])
        elif model_type == "two":
            # two models
            noised_predictions_band_alignn.append(
                noise_robust_dict["bandgap_onehot_alignn"]
            )
            noised_predictions_e_form_alignn.append(
                noise_robust_dict["ef_onehot_alignn"]
            )
            noised_predictions_band_crystalformer.append(
                noise_robust_dict["bandgap_onehot_crystalformer"]
            )
            noised_predictions_e_form_crystalformer.append(
                noise_robust_dict["ef_onehot_crystalformer"]
            )
        else:
            raise ValueError("model_type must be single or two")

    if model_type == "single":
        noised_predictions_band_array = torch.stack(noised_predictions_band).squeeze()
        band_mean = noised_predictions_band_array.mean(dim=0)
        band_std = noised_predictions_band_array.std(dim=0)
        noised_predictions_eform_array = torch.stack(
            noised_predictions_e_form
        ).squeeze()
        e_form_mean = noised_predictions_eform_array.mean(dim=0)
        e_form_std = noised_predictions_eform_array.std(dim=0)
        optimized_mini_batch_inputs_dict[f"band_mean_noise{noise_scale}"] = band_mean
        optimized_mini_batch_inputs_dict[f"band_std_noise{noise_scale}"] = band_std
        optimized_mini_batch_inputs_dict[f"e_form_mean_noise{noise_scale}"] = (
            e_form_mean
        )
        optimized_mini_batch_inputs_dict[f"e_form_std_noise{noise_scale}"] = e_form_std
    elif model_type == "two":
        noised_predictions_band_alignn_array = torch.stack(
            noised_predictions_band_alignn
        ).squeeze()
        band_mean_alignn = noised_predictions_band_alignn_array.mean(dim=0)
        band_std_alignn = noised_predictions_band_alignn_array.std(dim=0)
        noised_predictions_eform_alignn_array = torch.stack(
            noised_predictions_e_form_alignn
        ).squeeze()
        e_form_mean_alignn = noised_predictions_eform_alignn_array.mean(dim=0)
        e_form_std_alignn = noised_predictions_eform_alignn_array.std(dim=0)
        noised_predictions_band_crystalformer_array = torch.stack(
            noised_predictions_band_crystalformer
        ).squeeze()
        band_mean_crystalformer = noised_predictions_band_crystalformer_array.mean(
            dim=0
        )
        band_std_crystalformer = noised_predictions_band_crystalformer_array.std(dim=0)
        noised_predictions_eform_crystalformer_array = torch.stack(
            noised_predictions_e_form_crystalformer
        ).squeeze()
        e_form_mean_crystalformer = noised_predictions_eform_crystalformer_array.mean(
            dim=0
        )
        e_form_std_crystalformer = noised_predictions_eform_crystalformer_array.std(
            dim=0
        )
        optimized_mini_batch_inputs_dict[f"band_mean_alignn_noise{noise_scale}"] = (
            band_mean_alignn
        )
        optimized_mini_batch_inputs_dict[f"band_std_alignn_noise{noise_scale}"] = (
            band_std_alignn
        )
        optimized_mini_batch_inputs_dict[f"e_form_mean_alignn_noise{noise_scale}"] = (
            e_form_mean_alignn
        )
        optimized_mini_batch_inputs_dict[f"e_form_std_alignn_noise{noise_scale}"] = (
            e_form_std_alignn
        )
        optimized_mini_batch_inputs_dict[
            f"band_mean_crystalformer_noise{noise_scale}"
        ] = band_mean_crystalformer
        optimized_mini_batch_inputs_dict[
            f"band_std_crystalformer_noise{noise_scale}"
        ] = band_std_crystalformer
        optimized_mini_batch_inputs_dict[
            f"e_form_mean_crystalformer_noise{noise_scale}"
        ] = e_form_mean_crystalformer
        optimized_mini_batch_inputs_dict[
            f"e_form_std_crystalformer_noise{noise_scale}"
        ] = e_form_std_crystalformer

    return optimized_mini_batch_inputs_dict


# In[ ]:


def calculate_loss_from_output(
    bandgap_pred: torch.Tensor,
    ef_pred: torch.Tensor,
    sharpened_ox_mask: torch.Tensor,
    normalized_dist: torch.Tensor,
    site_ids: torch.Tensor,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    ef_coef: float,
    e_form_min: Optional[float],
    crystal_system: Optional[str],
    radii_tensor: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    gap_loss = gap_loss_func(bandgap_pred, reduction="none")
    if e_form_min is not None:
        ef_loss = torch.clip(ef_pred.squeeze() - e_form_min, min=0) * ef_coef
    else:
        ef_loss = ef_pred.squeeze() * ef_coef
    loss = gap_loss + ef_loss

    # tolerance loss
    if crystal_system == "perovskite":
        num_batch_crystal = len(gap_loss)  # number of crystals in a batch
        # calculate mean radii at each site normalized by the mean of oxides
        radii_per_atom = torch.sum(
            sharpened_ox_mask.unsqueeze(1) * radii_tensor, dim=2
        )  # shape=(N, num_type_of_atoms)
        site_mean_raddi = torch.sum(normalized_dist * radii_per_atom, dim=1).view(
            num_batch_crystal, -1
        )
        # calculate mean raddi at each crystal. A_site: 0, B_site: 1, X_site: 2
        A_site_mask = site_ids.view(num_batch_crystal, -1) == 0
        A_site_raddi = torch.where(
            A_site_mask,
            site_mean_raddi,
            torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device),
        ).sum(dim=1) / A_site_mask.sum(dim=1)
        B_site_mask = site_ids.view(num_batch_crystal, -1) == 1
        B_site_raddi = torch.where(
            B_site_mask,
            site_mean_raddi,
            torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device),
        ).sum(dim=1) / B_site_mask.sum(dim=1)
        X_site_mask = site_ids.view(num_batch_crystal, -1) == 2
        X_site_raddi = torch.where(
            X_site_mask,
            site_mean_raddi,
            torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device),
        ).sum(dim=1) / X_site_mask.sum(dim=1)
        # testing
        torch.testing.assert_close(
            A_site_mask.sum(dim=1), B_site_mask.sum(dim=1), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            A_site_mask.sum(dim=1) * 3, X_site_mask.sum(dim=1), atol=1e-5, rtol=1e-5
        )
        # calculate tolerance and its loss
        tolerance = (A_site_raddi + X_site_raddi) / (
            (2**0.5) * (B_site_raddi + X_site_raddi)
        )
        tolerance_loss = tolerance_loss_func(tolerance)
        loss = loss + tolerance_loss

    else:
        tolerance = torch.zeros_like(gap_loss).squeeze() * torch.nan
        tolerance_loss = torch.zeros_like(gap_loss).squeeze() * torch.nan

    total_each_loss = torch.nan_to_num(loss, nan=0.0)

    total_loss = total_each_loss.mean()
    return total_loss, total_each_loss, gap_loss, ef_loss, tolerance_loss, tolerance


# normalized_dist = optimized_mini_batch_inputs_dict['normalized_dist']
# radii_tensor = optimized_mini_batch_inputs_dict['radii_tensor']
# sharpened_ox_mask = optimized_mini_batch_inputs_dict['sharpened_ox_mask']
# site_ids = optimized_mini_batch_inputs_dict['site_ids'].to('cuda')
# batch = optimized_mini_batch_inputs_dict['batch']
# num_batch_crystal = 16
# radii_per_atom = torch.sum(sharpened_ox_mask.unsqueeze(1) * radii_tensor, dim=2) # shape=(N, num_type_of_atoms)
# site_mean_raddi = torch.sum(normalized_dist * radii_per_atom, dim=1).view(num_batch_crystal, -1)
# A_site_mask = site_ids.view(num_batch_crystal, -1) == 0
# A_site_raddi = torch.where(A_site_mask, site_mean_raddi , torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device)).sum(dim=1) / A_site_mask.sum(dim=1)
# B_site_mask = site_ids.view(num_batch_crystal, -1) == 1
# B_site_raddi = torch.where(B_site_mask, site_mean_raddi , torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device)).sum(dim=1) / B_site_mask.sum(dim=1)
# X_site_mask = site_ids.view(num_batch_crystal, -1) == 2
# X_site_raddi = torch.where(X_site_mask, site_mean_raddi , torch.zeros_like(site_mean_raddi, device=site_mean_raddi.device)).sum(dim=1) / X_site_mask.sum(dim=1)
# tolerance = (A_site_raddi + X_site_raddi) / ((2 ** 0.5) * (B_site_raddi + X_site_raddi))

# In[ ]:


# In[ ]:


# In[ ]:


def define_optimizer_and_scheduler(
    lattice_lr: float,
    atom_lr: float,
    coords_lr: float,
    ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor],
    atomic_distribution: torch.Tensor,
    scaled_batch_abc: torch.Tensor,
    scaled_batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    angle_optimization: bool,
    num_steps: int,
    lattice_cycle: int,
    atom_cycle: int,
    coords_cycle: int,
):
    # optimizer and scheduler
    if ox_mask_learnable_tensor_per_crystal is not None:
        print("<info> atomic_mask is used")
        optimizer_atom = torch.optim.Adam(
            [atomic_distribution, ox_mask_learnable_tensor_per_crystal], lr=atom_lr
        )
    else:
        optimizer_atom = torch.optim.Adam([atomic_distribution], lr=atom_lr)

    if angle_optimization:
        print("<info> angle optimization is enabled")
        optimizer_lattice = torch.optim.Adam(
            [scaled_batch_abc, scaled_batch_angle], lr=lattice_lr
        )
    else:
        print("<info> angle optimization is disabled")
        optimizer_lattice = torch.optim.Adam([scaled_batch_abc], lr=lattice_lr)

    optimizer_coords = torch.optim.Adam([batch_dir_coords], lr=coords_lr)
    scheduler_lattice = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_lattice, T_max=num_steps // lattice_cycle, eta_min=0
    )
    scheduler_atom = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_atom, T_max=num_steps // atom_cycle, eta_min=0
    )
    scheduler_coords = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_coords, T_max=num_steps // coords_cycle, eta_min=0
    )

    return (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    )


# In[ ]:


def calculate_onehot(
    output_dict: Dict[str, torch.Tensor],
    atom_feat_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    原子分布をonehotに変換する
    """
    max_val, max_index = torch.max(output_dict["normalized_dist"], dim=1)
    onehot_x = torch.zeros(
        output_dict["normalized_dist"].shape,
        device=output_dict["normalized_dist"].device,
    )
    onehot_x[torch.arange(output_dict["normalized_dist"].shape[0]), max_index] = 1
    assert (torch.max(onehot_x, dim=1)[1] == max_index).all()

    _, max_ox_index = torch.max(output_dict["sharpened_ox_mask"], dim=1)
    onehot_ox_mask = torch.zeros_like(output_dict["sharpened_ox_mask"])
    onehot_ox_mask[
        torch.arange(output_dict["sharpened_ox_mask"].shape[0]), max_ox_index
    ] = 1
    assert (torch.max(onehot_ox_mask, dim=1)[1] == max_ox_index).all()

    if atom_feat_matrix is not None:
        onehot_atom_feat = torch.matmul(onehot_x, atom_feat_matrix)
    else:
        onehot_atom_feat = torch.zeros_like(onehot_x) * torch.nan
    return max_val, onehot_x, onehot_atom_feat, onehot_ox_mask


# In[ ]:


# ### ALIGNNの最適化と評価関数

# In[ ]:


def ALIGNN_prediction(
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    g: dgl.DGLGraph,
    lg: dgl.DGLGraph,
    atomic_features: torch.Tensor,
    bondlength: torch.Tensor,
    bond_angles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bandgap_pred = torch.clip(
        bandgap_model((g, lg), atomic_features, bondlength, bond_angles), min=0
    )
    ef_pred = e_form_model((g, lg), atomic_features, bondlength, bond_angles)
    return bandgap_pred, ef_pred


# In[ ]:


def forward_propagation_ALIGNN(
    optimization_targets: List[torch.Tensor],
    fixed_inputs: Dict[str, torch.Tensor],
    scalers: dict,
    temperature: float,
    limit_coords_displacement: Optional[float],
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    adding_noise_scale: Optional[float],
    value_round: Optional[int] = None,
):
    (
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ) = optimization_targets
    batch_dst_ids = fixed_inputs["batch_dst_ids"]
    batch_src_ids = fixed_inputs["batch_src_ids"]
    batch_displace = fixed_inputs["batch_displace"]
    z_src_ids = fixed_inputs["z_src_ids"]
    z_dst_ids = fixed_inputs["z_dst_ids"]
    size = fixed_inputs["size"]
    num_edges = fixed_inputs["num_edges"]
    atom_feat_matrix = fixed_inputs["atom_feat_matrix"]
    atomic_mask = fixed_inputs["atomic_mask"]
    ox_states_used_mask = fixed_inputs["ox_states_used_mask"]
    g = fixed_inputs["g"]
    lg = fixed_inputs["lg"]

    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]

    (
        atomic_features,
        bondlength,
        bond_angles,
        batch_abc,
        batch_angle,
        lattice_vectors,
        normalized_dist,
        normed_batch_dir_coords,
        sharpened_ox_mask,
    ) = ALIGNN_preprocess(
        scaled_batch_abc=scaled_batch_abc.to("cuda"),
        scaled_batch_angle=scaled_batch_angle.to("cuda"),
        batch_dir_coords=batch_dir_coords.to("cuda"),
        atomic_distribution=atomic_distribution.to("cuda"),
        batch_dst_ids=batch_dst_ids.to("cuda"),
        batch_src_ids=batch_src_ids.to("cuda"),
        batch_displace=batch_displace.to("cuda"),
        z_src_ids=z_src_ids.to("cuda"),
        z_dst_ids=z_dst_ids.to("cuda"),
        size=size.to("cuda"),
        init_coords=fixed_inputs["init_coords"].to("cuda"),
        temperature=temperature,
        abc_scaler=abc_scaler,
        angle_scaler=angle_scaler,
        limit_coords_displacement=limit_coords_displacement,
        num_edges=num_edges.to("cuda"),
        adding_noise_scale=adding_noise_scale,
        atom_feat_matrix=atom_feat_matrix.to("cuda"),
        atomic_mask=atomic_mask.to("cuda"),
        ox_states_used_mask=ox_states_used_mask.to("cuda"),
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal.to(
            "cuda"
        ),
    )

    bandgap_pred, ef_pred = ALIGNN_prediction(
        bandgap_model=bandgap_model,
        e_form_model=e_form_model,
        g=g.to("cuda"),
        lg=lg.to("cuda"),
        atomic_features=atomic_features.to("cuda"),
        bondlength=bondlength.to("cuda"),
        bond_angles=bond_angles.to("cuda"),
    )

    output_dict = {
        "bandgap_pred": bandgap_pred,
        "ef_pred": ef_pred,
        "atomic_features": atomic_features,
        "bondlength": bondlength,
        "bond_angles": bond_angles,
        "batch_abc": batch_abc,
        "batch_angle": batch_angle,
        "lattice_vectors": lattice_vectors,
        "normalized_dist": normalized_dist,
        "normed_batch_dir_coords": normed_batch_dir_coords,
        "sharpened_ox_mask": sharpened_ox_mask,
    }
    return bandgap_pred, ef_pred, output_dict


# In[ ]:


# In[ ]:


# In[ ]:


def optimize_solution_ALIGNN(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: Dict[str, Any],
    learning_rates: List[float],
    learning_rate_cycle: List[int],
    num_steps: int,
    dist_temp_sche: List[float],
    Ef_coef_schedule: List[float],
    step_of_x_opt_stop: int,
    gap_loss_func: torch.nn.Module,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    scalers: dict,
    adding_noise_scale: Optional[float],
    e_form_min: Optional[float],
    limit_coords_displacement: Optional[float],
    crystal_system: Optional[str] = None,
    angle_optimization: bool = True,
    tolerance_loss_func=Optional[torch.nn.Module],
    device: str = "cuda",
    time_series_save: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    結晶の最適化を行う関数
    Args:
        mini_batch_inputs_dict: ALIGNNへの入力を構成するために必要な変数
            atomic_distribution(torch.Tensor): 原子分布, 学習可能パラメータ
            batch_abc(torch.Tensor): 結晶軸長さ, 学習可能パラメータ
            batch_angle(torch.Tensor): 結晶軸角度, 学習可能パラメータ
            num_edges(torch.Tensor): 結晶ごとのエッジ数, len(num_edges) == num_candidates
            batch_dir_coords(torch.Tensor): バッチ内部に存在する全原子の内部座標
            batch_dst_ids(torch.Tensor): バッチ内部に存在する全結合の終点のID
            batch_src_ids(torch.Tensor): バッチ内部に存在する全結合の始点のID
            batch_displace(torch.Tensor): バッチ内部に存在する全結合のdisplacement
            z_src_ids(torch.Tensor): ライングラフのエッジの始点のID
            z_dst_ids(torch.Tensor): ライングラフのエッジの終点のID
        num_steps(int): 最適化のステップ数
        dist_temp_sche(List[float]): Softmax温度スケジュール
        Ef_coef_schedule(List[float]): Ef損失の導入スケジュール
        step_of_x_opt_stop(int): xの最適化を止めるステップ数
        bandgap_model(torch.nn.Module): bandgapを予測するモデル
        e_form_model(torch.nn.Module): Efを予測するモデル
    Outputs:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: 最適化後の変数、bandgapの損失の履歴、Efの損失の履歴
    """
    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]
    atomic_distribution = nn.Parameter(
        mini_batch_inputs_dict["atomic_distribution"].clone().to("cuda"),
        requires_grad=True,
    )
    scaled_batch_abc = nn.Parameter(
        abc_scaler.scale(mini_batch_inputs_dict["batch_abc"].clone().to("cuda")),
        requires_grad=True,
    )
    scaled_batch_angle = nn.Parameter(
        angle_scaler.scale(mini_batch_inputs_dict["batch_angle"].clone().to("cuda")),
        requires_grad=True,
    )
    ox_mask_learnable_tensor_per_crystal = nn.Parameter(
        mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"]
        .to("cuda")
        .clone(),
        requires_grad=True,
    )
    batch_dir_coords = nn.Parameter(
        mini_batch_inputs_dict["batch_dir_coords"].to("cuda").clone(),
        requires_grad=True,
    )
    radii_tensor = mini_batch_inputs_dict["radii_tensor"].to("cuda")
    site_ids = mini_batch_inputs_dict["site_ids"].to("cuda")

    # optimizer and scheduler
    (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    ) = define_optimizer_and_scheduler(
        lattice_lr=learning_rates[0],
        atom_lr=learning_rates[1],
        coords_lr=learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=atomic_distribution,
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        angle_optimization=angle_optimization,
        num_steps=num_steps,
        lattice_cycle=learning_rate_cycle[0],
        atom_cycle=learning_rate_cycle[1],
        coords_cycle=learning_rate_cycle[2],
    )

    # initialzation of history
    lattice_mat_history, coords_history, normed_coords_history, elememts_history = (
        [],
        [],
        [],
        [],
    )
    abc_history, angle_history = [], []
    gap_history, e_form_history, t_loss_history = [], [], []
    VALID_ELEMENTS98 = create_element_list()
    lattice_lr_history, atom_lr_history, coords_lr_history = [], [], []

    # (English) After copy_and_mutate, it is necessary to perform the ALIGNN graph update_step.
    optimize_mode = settings_dict["optimize_mode"]
    parse_evolution_parameters_dict = parse_evolution_parameters(
        optimize_mode.split("sa___")[0]
    )
    if parse_evolution_parameters_dict["steps_cm"][0] != "None":
        additional_alignn_update_steps = [
            int(step) + 1 for step in parse_evolution_parameters_dict["steps_cm"]
        ]
        settings_dict["ALIGNN_update_step"].extend(additional_alignn_update_steps)

    optimization_targets = [
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ]
    optimization_target_names = [
        "scaled_batch_abc",
        "scaled_batch_angle",
        "batch_dir_coords",
        "atomic_distribution",
        "ox_mask_learnable_tensor_per_crystal",
    ]
    for step_i in tqdm(range(num_steps), desc="Optimizing", leave=False):
        ### Update graph and inputs
        if step_i in settings_dict["ALIGNN_update_step"]:
            print(f"<info> graph update at step {step_i}")
            # 学習パラメータとOptimizerを再度初期化する。
            #  cosine annealingのlearning rate の最大値を、現在の値へと更新する
            lattice_lr = optimizer_lattice.param_groups[0]["lr"]
            atom_lr = optimizer_atom.param_groups[0]["lr"]
            coords_lr = optimizer_coords.param_groups[0]["lr"]
            settings_dict["learning_rates"] = [lattice_lr, atom_lr, coords_lr]

            (
                upatate_mini_batch_inputs_dict,
                updated_scalers,
                optimiers_dict,
                learnable_parameters_dict,
            ) = update_graphs_and_trainable_parameters(
                optimized_mini_batch_inputs_dict={
                    "atomic_distribution": atomic_distribution,
                    "batch_abc": abc_scaler.rescale(scaled_batch_abc),
                    "batch_angle": angle_scaler.rescale(scaled_batch_angle),
                    "batch_dir_coords": batch_dir_coords,
                    "ox_mask_learnable_tensor_per_crystal": ox_mask_learnable_tensor_per_crystal,
                    "size": mini_batch_inputs_dict["size"],
                    "atomic_mask": mini_batch_inputs_dict["atomic_mask"],
                    "ox_states_used_mask": mini_batch_inputs_dict[
                        "ox_states_used_mask"
                    ],
                    "atom_feat_matrix": mini_batch_inputs_dict["atom_feat_matrix"],
                    "site_ids": mini_batch_inputs_dict["site_ids"],
                    "radii_tensor": mini_batch_inputs_dict["radii_tensor"],
                    "init_coords": mini_batch_inputs_dict["init_coords"],
                    "scaled_batch_abc": scaled_batch_abc,
                    "scaled_batch_angle": scaled_batch_angle,
                    "fnames": mini_batch_inputs_dict["fnames"],
                    "g": mini_batch_inputs_dict["g"],
                },
                settings_dict=settings_dict,
                tmp_poscar_dir="./tmp_poscars",
            )
            del mini_batch_inputs_dict, scalers
            mini_batch_inputs_dict = upatate_mini_batch_inputs_dict
            scalers = updated_scalers
            abc_scaler, angle_scaler = scalers["abc_scaler"], scalers["angle_scaler"]
            ##### update variables
            scaled_batch_abc, scaled_batch_angle, batch_dir_coords = (
                learnable_parameters_dict["scaled_batch_abc"],
                learnable_parameters_dict["scaled_batch_angle"],
                learnable_parameters_dict["batch_dir_coords"],
            )
            atomic_distribution, ox_mask_learnable_tensor_per_crystal = (
                learnable_parameters_dict["atomic_distribution"],
                learnable_parameters_dict["ox_mask_learnable_tensor_per_crystal"],
            )
            optimizer_lattice, optimizer_atom, optimizer_coords = (
                optimiers_dict["optimizer_lattice"],
                optimiers_dict["optimizer_atom"],
                optimiers_dict["optimizer_coords"],
            )
            scheduler_lattice, scheduler_atom, scheduler_coords = (
                optimiers_dict["scheduler_lattice"],
                optimiers_dict["scheduler_atom"],
                optimiers_dict["scheduler_coords"],
            )
            optimization_targets = [
                scaled_batch_abc,
                scaled_batch_angle,
                batch_dir_coords,
                atomic_distribution,
                ox_mask_learnable_tensor_per_crystal,
            ]

        ### backpropのたびに計算する
        # re-calculate atomic features, crystal structures, bondlength, bond angles
        bandgap_pred, ef_pred, output_dict = forward_propagation_ALIGNN(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=dist_temp_sche[step_i],
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
        )
        total_loss, each_loss, gap_loss, ef_loss, tolerance_loss, tolerance = (
            calculate_loss_from_output(
                bandgap_pred=bandgap_pred,
                ef_pred=ef_pred,
                sharpened_ox_mask=output_dict["sharpened_ox_mask"],
                normalized_dist=output_dict["normalized_dist"],
                site_ids=site_ids,
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                ef_coef=Ef_coef_schedule[step_i],
                crystal_system=crystal_system,
                e_form_min=e_form_min,
                radii_tensor=radii_tensor,
            )
        )

        if step_i == 0:
            # save initial poscar
            save_poscar(mini_batch_inputs_dict | output_dict, "init_poscars")

        # Update parameters
        size = mini_batch_inputs_dict["size"].to("cuda")
        cry_atom_data = torch.repeat_interleave(
            torch.arange(len(size), device=size.device), size
        )
        total_loss, optimization_targets, additional_info = update_parameters_step(
            optimization_targets=optimization_targets,
            optimization_target_names=optimization_target_names,
            step_i=step_i,
            total_loss=total_loss,
            gap_loss=gap_loss,
            ef_loss=ef_loss,
            tolerance_loss=tolerance_loss,
            each_loss=each_loss,
            total_steps=num_steps,
            learning_rates=learning_rates,
            cry_atom_data=cry_atom_data,
            size=size,
            atomic_mask=mini_batch_inputs_dict["atomic_mask"],
            ox_states_used_mask=mini_batch_inputs_dict["ox_states_used_mask"],
            site_ids=mini_batch_inputs_dict["site_ids"].to("cuda"),
            radii_tensor=mini_batch_inputs_dict["radii_tensor"].to("cuda"),
            init_coords=mini_batch_inputs_dict["init_coords"].to("cuda"),
            step_of_x_opt_stop=step_of_x_opt_stop,
            optimize_mode=optimize_mode,
            angle_optimization=angle_optimization,
            optimizer_lattice=optimizer_lattice,
            optimizer_atom=optimizer_atom,
            optimizer_coords=optimizer_coords,
            scheduler_lattice=scheduler_lattice,
            scheduler_atom=scheduler_atom,
            scheduler_coords=scheduler_coords,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            limit_coords_displacement=limit_coords_displacement,
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            Ef_coef_value=Ef_coef_schedule[step_i],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
        )
        # update optimizer
        optimizer_atom = additional_info["optimizer_atom"]
        optimizer_lattice = additional_info["optimizer_lattice"]
        optimizer_coords = additional_info["optimizer_coords"]
        scheduler_lattice = additional_info["scheduler_lattice"]
        scheduler_atom = additional_info["scheduler_atom"]
        scheduler_coords = additional_info["scheduler_coords"]

        mini_batch_inputs_dict["batch"] = additional_info["new_cry_atom_data"].to(
            "cuda"
        )
        mini_batch_inputs_dict["size"] = additional_info["new_size"].to("cuda")
        mini_batch_inputs_dict["atomic_mask"] = additional_info["atomic_mask"].to(
            "cuda"
        )
        mini_batch_inputs_dict["ox_states_used_mask"] = additional_info[
            "ox_states_used_mask"
        ].to("cuda")
        mini_batch_inputs_dict["site_ids"] = additional_info["site_ids"].to("cuda")
        mini_batch_inputs_dict["radii_tensor"] = additional_info["radii_tensor"].to(
            "cuda"
        )
        mini_batch_inputs_dict["init_coords"] = additional_info["init_coords"].to(
            "cuda"
        )

        scalers["abc_scaler"] = additional_info["abc_scaler"]
        abc_scaler = additional_info["abc_scaler"]
        # 損失の記録
        gap_history.append(bandgap_pred.detach().cpu().numpy())
        e_form_history.append(ef_pred.detach().cpu().numpy())
        t_loss_history.append(tolerance.detach().cpu().numpy())

        # learning rateの記録
        lattice_lr_history.append(optimizer_lattice.param_groups[0]["lr"])
        atom_lr_history.append(optimizer_atom.param_groups[0]["lr"])
        coords_lr_history.append(optimizer_coords.param_groups[0]["lr"])
        # print(f'step{step_i}. coords_lr:{optimizer_coords.param_groups[0]["lr"]}, lattice_lr:{optimizer_lattice.param_groups[0]["lr"]}, atom_lr:{optimizer_atom.param_groups[0]["lr"]}')

        if time_series_save:
            lattice_mat_history.append(
                output_dict["lattice_vectors"].detach().cpu().numpy()
            )
            normed_coords_history.append(
                output_dict["normed_batch_dir_coords"].detach().cpu().numpy()
            )
            coords_history.append(batch_dir_coords.detach().cpu().numpy())
            abc_history.append(output_dict["batch_abc"].detach().cpu().numpy())
            angle_history.append(output_dict["batch_angle"].detach().cpu().numpy())
            elements = [
                VALID_ELEMENTS98[atom_idx]
                for atom_idx in torch.argmax(output_dict["normalized_dist"], dim=1)
            ]
            elememts_history.append(elements)

    # End of optimization
    [
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ] = optimization_targets

    optimized_mini_batch_inputs_dict = mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict["atomic_distribution"] = atomic_distribution
    optimized_mini_batch_inputs_dict["scaled_batch_abc"] = scaled_batch_abc
    optimized_mini_batch_inputs_dict["scaled_batch_angle"] = scaled_batch_angle
    optimized_mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
        ox_mask_learnable_tensor_per_crystal
    )
    optimized_mini_batch_inputs_dict["batch_dir_coords"] = batch_dir_coords
    optimized_mini_batch_inputs_dict["gap_history"] = np.transpose(
        np.array(gap_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["ef_history"] = np.transpose(
        np.array(e_form_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["t_history"] = np.transpose(
        np.array(t_loss_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["lattice_lr_history"] = lattice_lr_history
    optimized_mini_batch_inputs_dict["atom_lr_history"] = atom_lr_history
    optimized_mini_batch_inputs_dict["coords_lr_history"] = coords_lr_history
    time_series = [
        lattice_mat_history,
        coords_history,
        normed_coords_history,
        abc_history,
        angle_history,
        elememts_history,
    ]

    return (optimized_mini_batch_inputs_dict, time_series, scalers)


# In[ ]:


def evaluation_for_each_batch_ALIGNN(
    settings_dict: Dict[str, Any],
    optimized_mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    crystal_system: Optional[str],
    limit_coords_displacement: Optional[float],
    e_form_min: Optional[float],
    Ef_coef_schedule: List[float],
    adding_noise_scale: Optional[float],
    onehot_temperature: float = 1e-8,
):
    """
    結晶の最適化を行う関数
    Args:
        optimized_mini_batch_inputs_dict: ALIGNNへの入力を構成するために必要な変数(一部最適化後)
            atomic_distribution(torch.Tensor): 原子分布, 学習可能パラメータ
            batch_abc(torch.Tensor): 結晶軸長さ, 学習可能パラメータ
            batch_angle(torch.Tensor): 結晶軸角度, 学習可能パラメータ
            num_edges(torch.Tensor): 結晶ごとのエッジ数, len(num_edges) == num_candidates
            batch_dir_coords(torch.Tensor): バッチ内部に存在する全原子の内部座標
            batch_dst_ids(torch.Tensor): バッチ内部に存在する全結合の終点のID
            batch_src_ids(torch.Tensor): バッチ内部に存在する全結合の始点のID
            batch_displace(torch.Tensor): バッチ内部に存在する全結合のdisplacement
            z_src_ids(torch.Tensor): ライングラフのエッジの始点のID
            z_dst_ids(torch.Tensor): ライングラフのエッジの終点のID
        evaluation_result_list: 評価結果のリスト
        history_list: 履歴のリスト
        scalers: データのスケーラー, [abc_scaler, angle_scaler]
    """

    print("<info> graph update at evaluation stage")
    updated_dict, scalers, optimiers_dict, learnable_parameters_dict = (
        update_graphs_and_trainable_parameters(
            optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
            settings_dict=settings_dict,
            tmp_poscar_dir="./tmp_poscars",
        )
    )
    optimized_mini_batch_inputs_dict.update(updated_dict)

    abc_scaler, angle_scaler = scalers["abc_scaler"], scalers["angle_scaler"]

    g = optimized_mini_batch_inputs_dict["g"].to("cuda")
    lg = optimized_mini_batch_inputs_dict["lg"].to("cuda")
    atom_feat_matrix = optimized_mini_batch_inputs_dict["atom_feat_matrix"].to("cuda")
    radii_tensor = optimized_mini_batch_inputs_dict["radii_tensor"].to("cuda")

    # print("site_ids in eval:", optimized_mini_batch_inputs_dict['site_ids'])

    with torch.no_grad():
        atomic_distribution = optimized_mini_batch_inputs_dict["atomic_distribution"]
        # scaled_batch_abc = optimized_mini_batch_inputs_dict['scaled_batch_abc']
        # scaled_batch_angle = optimized_mini_batch_inputs_dict['scaled_batch_angle']
        # batch_dir_coords = optimized_mini_batch_inputs_dict['batch_dir_coords'] #あとで消す??
        scaled_batch_abc = torch.round(
            optimized_mini_batch_inputs_dict["scaled_batch_abc"], decimals=4
        )
        scaled_batch_angle = torch.round(
            optimized_mini_batch_inputs_dict["scaled_batch_angle"], decimals=4
        )
        batch_dir_coords = torch.round(
            optimized_mini_batch_inputs_dict["batch_dir_coords"], decimals=4
        )
        ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict[
            "ox_mask_learnable_tensor_per_crystal"
        ]
        optimization_targets = [
            scaled_batch_abc,
            scaled_batch_angle,
            batch_dir_coords,
            atomic_distribution,
            ox_mask_learnable_tensor_per_crystal,
        ]

        # calculation with atomic distribution
        bandgap_dist, ef_dist, output_dict = forward_propagation_ALIGNN(
            optimization_targets=optimization_targets,
            fixed_inputs=optimized_mini_batch_inputs_dict,
            scalers=scalers,
            temperature=onehot_temperature,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
        )
        _, _, loss_dist, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_dist,
            ef_pred=ef_dist,
            sharpened_ox_mask=output_dict["sharpened_ox_mask"],
            normalized_dist=output_dict["normalized_dist"],
            site_ids=optimized_mini_batch_inputs_dict["site_ids"].to("cuda"),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[-1],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

        # calculation with one-hot atomic distribution
        max_val, onehot_x, onehot_atom_feat, onehot_ox_mask = calculate_onehot(
            output_dict, atom_feat_matrix
        )
        bandgap_onehot, ef_onehot = ALIGNN_prediction(
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            g=g.to("cuda"),
            lg=lg.to("cuda"),
            atomic_features=onehot_atom_feat,  # one-hot atom features
            bondlength=output_dict["bondlength"],
            bond_angles=output_dict["bond_angles"],
        )
        _, _, loss_onehot, _, tolerance_loss, tolerance = calculate_loss_from_output(
            bandgap_pred=bandgap_onehot,
            ef_pred=ef_onehot,
            sharpened_ox_mask=onehot_ox_mask,
            normalized_dist=onehot_x,
            site_ids=optimized_mini_batch_inputs_dict["site_ids"].to("cuda"),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[-1],
            e_form_min=e_form_min,
            crystal_system=crystal_system,
            radii_tensor=radii_tensor,
        )
    # あとで消す
    # print("onehot_atom_feat.shape", onehot_atom_feat.shape)
    # print("onehot_atom_feat", onehot_atom_feat)
    VALID_ELEMENTS98 = create_element_list()
    print(
        [
            VALID_ELEMENTS98[a_idx]
            for a_idx in np.argmax(onehot_x[-9:].detach().cpu().numpy(), axis=1)
        ]
    )
    print(batch_dir_coords[-9:].detach().cpu().numpy())

    # 結果の記録
    optimized_mini_batch_inputs_dict = optimized_mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict["max_val"] = max_val
    optimized_mini_batch_inputs_dict["bandgap_onehot"] = bandgap_onehot
    optimized_mini_batch_inputs_dict["bandgap_dist"] = bandgap_dist
    optimized_mini_batch_inputs_dict["ef_onehot"] = ef_onehot
    optimized_mini_batch_inputs_dict["ef_dist"] = ef_dist
    optimized_mini_batch_inputs_dict["loss_onehot"] = loss_onehot
    optimized_mini_batch_inputs_dict["loss_dist"] = loss_dist
    optimized_mini_batch_inputs_dict["onehot_x"] = onehot_x
    optimized_mini_batch_inputs_dict["tolerance"] = tolerance
    optimized_mini_batch_inputs_dict["tolerance_loss"] = tolerance_loss
    optimized_mini_batch_inputs_dict["onehot_ox_mask"] = onehot_ox_mask

    return optimized_mini_batch_inputs_dict


# #### ALIGNNのグラフを更新する関数

# In[ ]:


def matching_by_hash_table(a: np.ndarray, b: np.ndarray) -> List[int]:
    """A function to find the indices where each row of a and b match"""
    # ハッシュテーブルを作成
    hash_map = {tuple(row): idx for idx, row in enumerate(a)}

    # bの各行に対して対応するaのインデックスを見つける
    matching_indices = [hash_map[tuple(row)] for row in b]
    return matching_indices


def get_alignment_ids(
    fnames: np.ndarray,
    batch_abc: torch.Tensor,
    batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    atomic_distribution: torch.Tensor,
    size: torch.Tensor,
    updated_fnames: np.ndarray,
    updated_batch_abc: np.ndarray,
    updated_batch_dir_coords: np.ndarray,
    updated_size: np.ndarray,
):
    """
    1度POSCARを作成してグラフに変換すると、原子の位置が変わる。そのため、元のPOSCARの順番と、最適化後の順番が一致させるためのID配列を作成する。
    English:Once the POSCAR is created and converted to a graph, the positions of the atoms change. Therefore, an ID array is created to match the order of the original POSCAR and the optimized order.
    """

    # Get lattice alignment id array
    ### all fnames are unique
    assert len(fnames) == np.unique(fnames).shape[0]
    assert len(updated_fnames) == np.unique(updated_fnames).shape[0]
    # Convert the index of the old data to the index of the new data
    lattice_alignment_array = np.array(
        [list(fnames).index(fn) for fn in updated_fnames]
    )
    ### Check if the order of the data is the same
    assert (fnames[lattice_alignment_array] == updated_fnames).all()
    ### Check if the data is the same
    # nan_data_bool = ~torch.isnan(batch_abc[lattice_alignment_array].detach().cpu()).any(axis=1)
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle)
    assert lattice_vectors.shape == torch.Size([len(batch_abc), 3, 3])
    nan_data_bool = (torch.isnan(lattice_vectors).sum(dim=[1, 2]) == 0).detach().cpu()
    torch.testing.assert_close(
        batch_abc[lattice_alignment_array][nan_data_bool].detach().cpu(),
        updated_batch_abc[nan_data_bool].detach().cpu(),
    )
    lattice_vectors = lattice_vectors.detach().cpu().numpy()
    assert (size[lattice_alignment_array].to(updated_size.device) == updated_size).all()

    # compute lattice vectors

    # get atom-wise alignment id
    coordinate_lattice_id = np.concatenate(
        [np.array([lattice_id] * num) for lattice_id, num in enumerate(size)]
    )
    aligned_size = size.detach().cpu().numpy()[lattice_alignment_array]
    aligned_coordinate_lattice_id = np.concatenate(
        [np.array([lattice_id] * num) for lattice_id, num in enumerate(aligned_size)]
    )
    ### get the index of the new data in the old data
    updated_batch_dir_coords = updated_batch_dir_coords.detach().cpu().numpy()
    ### Get the index to align the atomic order for each crystal)
    atom_wise_alignment_list = []
    no_nan_bool = np.ones(shape=(len(batch_dir_coords),), dtype=bool)
    for lattice_id in range(len(size)):
        ### get new lattice id
        aligned_lattice_id = lattice_alignment_array[lattice_id]
        ### get atom-wise lattice id
        aligned_atom_bool = aligned_coordinate_lattice_id == aligned_lattice_id
        atom_bool = coordinate_lattice_id == lattice_id
        ### get lattice data
        lattice_abc = batch_abc[lattice_id].detach().cpu().numpy()
        lattice_angle = batch_angle[lattice_id].detach().cpu().numpy()
        lattice_batch_dir_coords = batch_dir_coords[atom_bool].detach().cpu().numpy()
        lattice_atom_distribution = (
            atomic_distribution[atom_bool].detach().cpu().numpy()
        )
        if (
            np.isnan(lattice_vectors[lattice_id]).any()
            or np.isnan(lattice_batch_dir_coords).any()
            or np.isnan(lattice_abc).any()
            or np.isnan(lattice_angle).any()
            or np.isnan(lattice_atom_distribution).any()
        ):
            ### if nan exists, the alignment order is no longer valid
            print(f"lattice_id: {fnames[lattice_id]} has nan")
            index = np.arange(len(lattice_batch_dir_coords))
            no_nan_bool[atom_bool] = False
        else:
            ### get the index to align the atomic order
            index = np.array(
                matching_by_hash_table(
                    lattice_batch_dir_coords,
                    updated_batch_dir_coords[aligned_atom_bool],
                )
            )
        index = list(index + len(atom_wise_alignment_list))
        atom_wise_alignment_list.extend(index)

    atom_wise_alignment_array = np.array(atom_wise_alignment_list)

    assert (
        batch_dir_coords[atom_wise_alignment_array].detach().cpu().numpy()[no_nan_bool]
        == updated_batch_dir_coords[no_nan_bool]
    ).all()

    return lattice_alignment_array, atom_wise_alignment_array, no_nan_bool


def get_alignment_ids_from_current_structure(
    optimized_mini_batch_inputs_dict: Dict[str, Any],
    settings_dict: Dict[str, Any],
    tmp_poscar_dir: str = "./tmp_poscars",
):
    ### update graph data
    all_inputs_dict = load_initial_data(
        settings_dict=settings_dict,
        model_name="ALIGNN",
        dir_path=tmp_poscar_dir,
        intermid_dir_path=settings_dict["intermid_dir_path"],
        crystal_system=None,  # マスクを引き継ぐのでNoneにする。perovskiteのときに不具合が発生するため
        max_ox_states=settings_dict["max_ox_states"],
        num_candidate=settings_dict["num_batch_crystal"],
        batch_size=settings_dict["num_batch_crystal"],
        use_atomic_mask=settings_dict["use_atomic_mask"],
        mask_method=settings_dict["mask_method"],
        test_mode=settings_dict["test_mode"],
        angle_range=settings_dict["angle_range"],
        abc_range=settings_dict["abc_range"],
        graph_update_mode=True,  # graph更新モード、原子マスクを作成しない
    )
    updated_minibatch_dict, updated_scalers = all_inputs_dict[0]

    # delete unnecessary data that is replaced by the original data
    del (
        updated_minibatch_dict["atomic_distribution"],
        updated_minibatch_dict["ox_mask_learnable_tensor_per_crystal"],
    )
    del (
        updated_minibatch_dict["ox_states_used_mask"],
        updated_minibatch_dict["atomic_mask"],
    )

    ### get alignment ids
    lattice_alignment_array, atom_wise_alignment_array, no_nan_bool = get_alignment_ids(
        fnames=optimized_mini_batch_inputs_dict["fnames"],
        batch_abc=optimized_mini_batch_inputs_dict["batch_abc"],
        batch_angle=optimized_mini_batch_inputs_dict["batch_angle"],
        batch_dir_coords=torch.remainder(
            optimized_mini_batch_inputs_dict["batch_dir_coords"], 1.0
        ),
        atomic_distribution=optimized_mini_batch_inputs_dict["atomic_distribution"],
        size=optimized_mini_batch_inputs_dict["size"].to("cuda"),
        updated_fnames=updated_minibatch_dict["fnames"],
        updated_batch_abc=updated_minibatch_dict["batch_abc"],
        updated_batch_dir_coords=updated_minibatch_dict["batch_dir_coords"],
        updated_size=updated_minibatch_dict["size"],
    )
    return (
        lattice_alignment_array,
        atom_wise_alignment_array,
        updated_minibatch_dict,
        updated_scalers,
        no_nan_bool,
    )


def update_graphs_and_trainable_parameters(
    optimized_mini_batch_inputs_dict: Dict[str, Any],
    settings_dict: Dict[str, Any],
    tmp_poscar_dir: str,
    onehot_temperature: float = 1e-8,
):
    """
    Updated tranable parameters
    """
    # get settings parameters
    learning_rates = settings_dict["learning_rates"]
    learning_rate_cycle = settings_dict["learning_rate_cycle"]
    num_steps = settings_dict["num_steps"]
    angle_optimization = settings_dict["angle_optimization"]
    print(
        " ----------------------------\n   <info> Update graph and trainable parameters...."
    )

    ### save current structures to poscar files
    save_poscar(optimized_mini_batch_inputs_dict, tmp_poscar_dir)

    # get alignment ids
    (
        lattice_alignment_array,
        atom_wise_alignment_array,
        updated_minibatch_dict,
        updated_scaler,
        no_nan_bool,
    ) = get_alignment_ids_from_current_structure(
        optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
        settings_dict=settings_dict,
        tmp_poscar_dir=tmp_poscar_dir,
    )
    # get data from optimized_mini_batch_inputs_dict
    atomic_distribution = optimized_mini_batch_inputs_dict["atomic_distribution"]
    ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict[
        "ox_mask_learnable_tensor_per_crystal"
    ]
    ox_states_used_mask = optimized_mini_batch_inputs_dict["ox_states_used_mask"]
    atomic_mask = optimized_mini_batch_inputs_dict["atomic_mask"]
    site_ids = optimized_mini_batch_inputs_dict["site_ids"]
    radii_tensor = optimized_mini_batch_inputs_dict["radii_tensor"]
    init_coords = optimized_mini_batch_inputs_dict["init_coords"]
    batch_dir_coords = optimized_mini_batch_inputs_dict[
        "batch_dir_coords"
    ]  # newly added, 検証のため追加した
    ox_states_used_mask = optimized_mini_batch_inputs_dict[
        "ox_states_used_mask"
    ]  # newly added, 検証のため追加した
    scaled_batch_abc = optimized_mini_batch_inputs_dict[
        "scaled_batch_abc"
    ]  # newly added, 検証のため追加した
    scaled_batch_angle = optimized_mini_batch_inputs_dict[
        "scaled_batch_angle"
    ]  # newly added, 検証のため追加した

    # align current atomic distribution for the updated structure
    updated_minibatch_dict["atomic_distribution"] = atomic_distribution[
        atom_wise_alignment_array
    ]
    updated_minibatch_dict["ox_mask_learnable_tensor_per_crystal"] = (
        ox_mask_learnable_tensor_per_crystal[lattice_alignment_array]
    )
    updated_minibatch_dict["ox_states_used_mask"] = ox_states_used_mask[
        lattice_alignment_array
    ]  # [atom_wise_alignment_array]
    updated_minibatch_dict["atomic_mask"] = atomic_mask[atom_wise_alignment_array]
    updated_minibatch_dict["site_ids"] = site_ids[atom_wise_alignment_array]
    updated_minibatch_dict["radii_tensor"] = radii_tensor[atom_wise_alignment_array]
    updated_minibatch_dict["init_coords"] = init_coords[atom_wise_alignment_array]
    updated_minibatch_dict["batch_dir_coords"] = batch_dir_coords[
        atom_wise_alignment_array
    ]
    updated_minibatch_dict["ox_states_used_mask"] = ox_states_used_mask[
        atom_wise_alignment_array
    ]
    updated_minibatch_dict["scaled_batch_abc"] = scaled_batch_abc[
        lattice_alignment_array
    ]
    updated_minibatch_dict["scaled_batch_angle"] = scaled_batch_angle[
        lattice_alignment_array
    ]

    updated_minibatch_dict["size"] = updated_minibatch_dict["size"].to("cuda")

    size = updated_minibatch_dict["size"]
    assert (
        updated_minibatch_dict["size"]
        == optimized_mini_batch_inputs_dict["size"][lattice_alignment_array]
    ).all()

    # get onehot atomic distribution
    ### masking and sharpening
    stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
        updated_minibatch_dict["ox_mask_learnable_tensor_per_crystal"]
        .clone()
        .detach()
        .cpu(),
        size,
    )
    atomic_distribution, sharpened_ox_mask = masking_atomic_distribution(
        updated_minibatch_dict["atomic_distribution"].clone().detach().cpu(),
        updated_minibatch_dict["atomic_mask"].clone().detach().cpu(),
        updated_minibatch_dict["ox_states_used_mask"].clone().detach().cpu(),
        stacked_learnable_ox_weight,
        onehot_temperature,
    )
    quasi_onehot_dist = temperature_softmax(
        atomic_distribution, temperature=onehot_temperature
    )
    ### get onehot atomic distribution
    _, onehot_x, _, _ = calculate_onehot(
        {
            "normalized_dist": quasi_onehot_dist,
            "sharpened_ox_mask": sharpened_ox_mask,
        },
        updated_minibatch_dict["atom_feat_matrix"],
    )

    ### Check if the elements match the updated structure
    VALID_ELEMENTS98 = create_element_list()
    updated_elements1 = [
        VALID_ELEMENTS98[a_idx - 1]
        for a_idx in updated_minibatch_dict["g"]
        .ndata["atomic_number"]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
        .reshape(-1)
    ]
    updated_elements2 = [
        VALID_ELEMENTS98[a_idx]
        for a_idx in torch.argmax(onehot_x, axis=1)
        .detach()
        .cpu()
        .numpy()
        .astype(int)
        .reshape(-1)
    ]
    assert (
        np.array(updated_elements1)[no_nan_bool]
        == np.array(updated_elements2)[no_nan_bool]
    ).all()

    # Get updated trainable parameters etc.
    abc_scaler = updated_scaler["abc_scaler"]
    angle_scaler = updated_scaler["angle_scaler"]
    atomic_distribution = nn.Parameter(
        updated_minibatch_dict["atomic_distribution"].clone().to("cuda"),
        requires_grad=True,
    )
    scaled_batch_abc = nn.Parameter(
        updated_minibatch_dict["scaled_batch_abc"].clone().to("cuda"),
        requires_grad=True,
    )
    scaled_batch_angle = nn.Parameter(
        updated_minibatch_dict["scaled_batch_angle"].clone().to("cuda"),
        requires_grad=True,
    )
    batch_dir_coords = nn.Parameter(
        updated_minibatch_dict["batch_dir_coords"].clone().to("cuda"),
        requires_grad=True,
    )
    ox_mask_learnable_tensor_per_crystal = nn.Parameter(
        updated_minibatch_dict["ox_mask_learnable_tensor_per_crystal"]
        .to("cuda")
        .clone(),
        requires_grad=True,
    )
    updated_minibatch_dict["radii_tensor"] = updated_minibatch_dict["radii_tensor"].to(
        "cuda"
    )
    updated_minibatch_dict["site_ids"] = updated_minibatch_dict["site_ids"].to("cuda")
    learnable_parameters_dict = {
        "scaled_batch_abc": scaled_batch_abc,
        "scaled_batch_angle": scaled_batch_angle,
        "batch_dir_coords": batch_dir_coords,
        "atomic_distribution": atomic_distribution,
        "ox_mask_learnable_tensor_per_crystal": ox_mask_learnable_tensor_per_crystal,
    }

    # optimizer and scheduler
    (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    ) = define_optimizer_and_scheduler(
        lattice_lr=learning_rates[0],
        atom_lr=learning_rates[1],
        coords_lr=learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=atomic_distribution,
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        angle_optimization=angle_optimization,
        num_steps=num_steps,
        lattice_cycle=learning_rate_cycle[0],
        atom_cycle=learning_rate_cycle[1],
        coords_cycle=learning_rate_cycle[2],
    )

    optimiers_dict = {
        "optimizer_lattice": optimizer_lattice,
        "optimizer_atom": optimizer_atom,
        "optimizer_coords": optimizer_coords,
        "scheduler_lattice": scheduler_lattice,
        "scheduler_atom": scheduler_atom,
        "scheduler_coords": scheduler_coords,
    }
    print(
        "    <info> End Updating graph and trainable parameters....\n ----------------------------"
    )

    return (
        updated_minibatch_dict,
        updated_scaler,
        optimiers_dict,
        learnable_parameters_dict,
    )


# In[ ]:


# In[ ]:


# In[ ]:


from typing import List, Tuple

import numpy as np
import torch


def calculate_update_steps(num_steps: int, num_update: int) -> List[int]:
    """Calculates the step indices where the sine values first exceed the specified thresholds.

    This function computes the sine value for a sequence of steps and finds the first step
    where the sine value exceeds each of a set of decreasing thresholds.

    Args:
        num_steps (int): The total number of steps to compute sine values for. Default is 200.
        num_update (int): The number of threshold values to compute, which are evenly spaced and decreasing. Default is 10.

    Returns:
        List[int]: A list of indices indicating the first step where the sine value exceeds each threshold.

    Raises:
        ValueError: If `num_update` is zero or negative, which would lead to division by zero.
    """
    if num_update <= 0:
        raise ValueError("num_update must be greater than 1 to avoid division by zero.")

    # Generate the step numbers
    steps = np.arange(num_steps)

    # Calculate the sine values
    sin_val_steps = np.sin(np.pi / 2 * steps / num_steps)

    # Calculate update thresholds
    update_thres = np.linspace(0, 1, num_update + 1, endpoint=False)[1:]

    # Find the first step that exceeds each threshold
    update_steps = []
    for thres in update_thres:
        idx = np.argmax(sin_val_steps > thres)
        update_steps.append(
            idx if sin_val_steps[idx] > thres else -1
        )  # -1 if no step exceeds the threshold

    return update_steps


# In[ ]:


# In[ ]:


# In[ ]:


# ### simulated_annearling

# In[ ]:


def parse_sa_parameters(optimize_mode_str: str) -> dict:
    """Parses an optimize_mode string to extract simulated annealing (SA) parameters.

    The optimize_mode string is expected to contain SA parameters after a 'sa___'
    delimiter. Individual settings are separated by '__', and each setting consists
    of a key and a value separated by '-' (the first '-' is used as the separator).
    If any parameter not defined in the default parameter list is encountered, a
    ValueError is raised.

    For example, an input string:
        "sa___mode--exponential__noise_scale-0.1"
    will produce a dictionary:
        {
            "mode": "exponential",
            "noise_scale": 0.1,
            "T0": 1.0,
            "T_min": 0.001,
            "alpha": 0.99,
            "beta": None,
            "c": 2.0,
        }

    Args:
        optimize_mode_str (str): The optimization mode string.

    Returns:
        dict: A dictionary of SA parameters.

    Raises:
        ValueError: If an unknown parameter is found in the input string.
    """
    # Default SA parameters
    defaults = {
        "mode": "exponential",  # Cooling schedule type: "exponential", "linear", "logarithmic"
        "noise_scale": 0.01,  # Scale for the noise added to the update
        "T0": 1.0,  # Initial temperature
        "T_min": 1e-6,  # Minimum temperature (lower bound)
        "alpha": 0.99,  # Cooling rate for exponential cooling
        "beta": None,  # Linear cooling decrement; if None, it will be computed automatically
        "c": 2.0,  # Offset for logarithmic cooling
    }

    # If the string starts with "sa", split by "___" to obtain the parameter part.
    if optimize_mode_str.startswith("sa"):
        parts = optimize_mode_str.split("___", 1)
        if len(parts) == 2:
            param_str = parts[1]
        else:
            # If only "sa" is provided, return the default parameters.
            return defaults
    else:
        # If "sa" is not included, assume no SA parameters are specified.
        return defaults

    # Split the parameters using '__' as the delimiter.
    settings = param_str.split("__")
    for setting in settings:
        # Each setting is expected to be in the format "key-value" (e.g., "mode--exponential" or "noise_scale-0.1")
        if "-" in setting:
            key, sep, value = setting.partition("-")
            key = key.strip()
            # Remove any extra '-' characters from the beginning of the value.
            value = value.lstrip("-").strip()
            # Check if the key is a valid parameter.
            if key not in defaults:
                raise ValueError(f"Unknown SA parameter: '{key}'")
            # Try to convert the value to an int or float if possible.
            if value.replace(".", "", 1).isdigit():
                value_converted = float(value) if "." in value else int(value)
            else:
                value_converted = value
            defaults[key] = value_converted
    defaults["noise_scale"] = float(defaults["noise_scale"])
    return defaults


# In[ ]:


def get_current_temperature(
    step_i: int,
    total_steps: int,
    schedule_type: str = "exponential",
    T0: float = 1.0,
    T_min: float = 1e-3,
    alpha: float = 0.99,
    beta: float = None,  # 線形冷却用。betaがNoneならT0とT_minから自動計算
    c: float = 2.0,  # 対数冷却用の定数
) -> float:
    """
    現在のステップにおける温度を、指定された冷却スケジュールに基づいて計算します。

    Args:
        step_i (int): 現在のステップ番号 (0-indexed)
        total_steps (int): 全体のステップ数
        schedule_type (str): 冷却スケジュールの種類。以下のいずれか:
            - "exponential": 指数冷却
            - "linear": 線形冷却
            - "logarithmic": 対数冷却
        T0 (float): 初期温度（step_i=0 のときの温度）
        T_min (float): 最小温度（最終ステップでの温度の下限）
        alpha (float): 指数冷却での冷却率（0 < alpha < 1）
        beta (float): 線形冷却での減衰量。Noneの場合、(T0 - T_min) / total_steps として計算
        c (float): 対数冷却でのオフセット。大きいほど冷却が緩やか

    Returns:
        float: 現在の温度
    """
    if schedule_type.lower() == "exponential":
        # 指数冷却： T = T0 * (alpha^step_i)
        T = T0 * (alpha**step_i)
        # 下限 T_min を下回らないようにする
        T = max(T, T_min)
    elif schedule_type.lower() == "linear":
        # 線形冷却： T = T0 - beta * step_i
        if beta is None:
            beta = (T0 - T_min) / total_steps
        T = T0 - beta * step_i
        T = max(T, T_min)
    elif schedule_type.lower() == "logarithmic":
        # 対数冷却： T = T0 / log(step_i + c)
        # step_iが0の場合は初期温度 T0 とする
        if step_i == 0:
            T = T0
        else:
            import math

            T = T0 / math.log(step_i + c)
            T = max(T, T_min)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    return T


# In[ ]:


import torch


def expand_accept_mask(
    accept_mask: torch.Tensor, cry_atom_data: torch.Tensor
) -> torch.Tensor:
    """Expands the accept_mask based on the counts in cry_atom_data.

    The function repeats each element of accept_mask according to the number of
    occurrences of its corresponding sample index in cry_atom_data.

    For example:
        If accept_mask = [0, 1, 0] and cry_atom_data = [0, 0, 0, 0, 1, 1, 2, 2, 2],
        then the resulting expanded mask will be:
            [0, 0, 0, 0, 1, 1, 0, 0, 0]
        (i.e. sample 0's flag repeated 4 times, sample 1's flag repeated 2 times,
         and sample 2's flag repeated 3 times).

    Args:
        accept_mask (torch.Tensor): 1D tensor of shape [N] containing the acceptance flags
            for N samples.
        cry_atom_data (torch.Tensor): 1D tensor containing sample indices for each atom.
            For instance, if sample 0 has 4 atoms, sample 1 has 2 atoms, and sample 2 has 3 atoms,
            cry_atom_data might be: tensor([0, 0, 0, 0, 1, 1, 2, 2, 2]).

    Returns:
        torch.Tensor: A 1D tensor where each element from accept_mask is repeated according
            to its count in cry_atom_data. The length of the output equals the number of elements
            in cry_atom_data.
    """

    assert (
        len(accept_mask) == max(cry_atom_data) + 1
    ), "accept_mask length must match the number of unique samples in cry_atom_data"
    # Initialize an empty list to collect the expanded mask values.
    expanded_list = []

    # Iterate over each sample index (assumed to be 0 to len(accept_mask)-1).
    for i in range(len(accept_mask)):
        # Count how many times sample i appears in cry_atom_data.
        count = int((cry_atom_data == i).sum().item())
        # Extend the result list by repeating accept_mask[i] 'count' times.
        expanded_list.extend([accept_mask[i].item()] * count)

    # Convert the expanded list back to a torch.Tensor.
    return torch.tensor(expanded_list, device=accept_mask.device)


# In[ ]:


import math
import random
from typing import Any, Dict, List, Tuple

import torch


def simulated_annearling_update(
    optimization_targets: list,
    step_i: int,
    total_loss: torch.Tensor,
    each_loss: torch.Tensor,  # shape: [batch_size]
    temp: float,
    learning_rates: List[float],  # [lr_lattice, lr_atom, lr_coords]
    sa_noise_scale: float,
    step_of_x_opt_stop: int,
    angle_optimization: bool,
    cry_atom_data: torch.Tensor,
    # 以下は候補損失評価に必要な引数
    mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    limit_coords_displacement: float,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    Ef_coef_value: float,
    crystal_system: torch.nn.Module,
    e_form_min: float,
    radii_tensor: torch.Tensor,
    abc_range: Tuple[float],
    angle_range: Tuple[float],
) -> Tuple[float, Dict[str, Any]]:
    """
    SAモードにおける1ステップ分の更新を実施する関数です。

    内部で以下を行います：
      1. 学習可能パラメータに対してzero_grad()を実施し、total_loss.backward() で勾配を計算
      2. 各パラメータの候補更新（勾配降下＋温度依存のノイズ付加）を計算し、候補状態に更新
      3. torch.no_grad()ブロック内で、候補状態でのフォワードパスと損失計算のロジックを流用して、各サンプルの候補損失（candidate_each_loss）を評価
      4. 各サンプルごとに受け入れ判定を行い、受け入れなかったサンプルは更新前のパラメータに復元
      5. 勾配をリセットし、更新後の平均損失と追加情報（受け入れマスクなど）を返します。

    Args:
        optimization_targets (list): 更新対象のパラメータ群（順に：
            [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal]）
        step_i (int): 現在のステップ番号
        total_loss (torch.Tensor): 総損失（スカラー）
        each_loss (torch.Tensor): 各サンプルの損失（shape: [batch_size]）
        temp (float): 現在の温度
        learning_rates (List[float]): 各グループの学習率 [lr_lattice, lr_atom, lr_coords]
        adding_noise_scale (float): SA更新時に加えるノイズのスケール
        step_of_x_opt_stop (int): atomパラメータ更新停止ステップ
        angle_optimization (bool): angle更新の有無
        mini_batch_inputs_dict, scalers, bandgap_model, e_form_model, limit_coords_displacement,
            gap_loss_func, tolerance_loss_func, Ef_coef_value, crystal_system, e_form_min, radii_tensor:
            候補状態での損失評価に必要な各種引数

    Returns:
        updated_loss (float): 更新後の平均損失（受け入れたサンプルは候補損失、棄却したサンプルは元の損失）
        additional_info (Dict[str, Any]): 追加情報（例:受け入れマスク、modeなど）
    """
    additional_info = {}

    # 1. 直接学習可能パラメータに対してzero_grad()を実施
    for param in optimization_targets:
        if param.grad is not None:
            param.grad.zero_()

    # 2. total_loss.backward() により勾配計算
    total_loss.backward()

    # 3. 保存：更新前のパラメータ（後で候補棄却時に戻すため）
    old_params = [param.detach().clone() for param in optimization_targets]

    # 4. 各パラメータの候補更新の計算
    lr_lattice = learning_rates[0]
    lr_atom = learning_rates[1]
    lr_coords = learning_rates[2]
    noise_scale = sa_noise_scale * temp

    # lattice更新 (scaled_batch_abc, scaled_batch_angle)
    candidate_scaled_batch_abc = (
        optimization_targets[0].detach()
        - lr_lattice * optimization_targets[0].grad
        + torch.randn_like(optimization_targets[0]) * noise_scale
    )
    if angle_optimization:
        candidate_scaled_batch_angle = (
            optimization_targets[1].detach()
            - lr_lattice * optimization_targets[1].grad
            + torch.randn_like(optimization_targets[1]) * noise_scale
        )
    else:
        candidate_scaled_batch_angle = optimization_targets[1].detach()

    # coords更新 (batch_dir_coords)
    candidate_batch_dir_coords = (
        optimization_targets[2].detach()
        - lr_coords * optimization_targets[2].grad
        + torch.randn_like(optimization_targets[2]) * noise_scale
    )

    # atom更新 (atomic_distribution, ox_mask_learnable_tensor_per_crystal)
    if step_i < step_of_x_opt_stop:
        candidate_atomic_distribution = (
            optimization_targets[3].detach()
            - lr_atom * optimization_targets[3].grad
            + torch.randn_like(optimization_targets[3]) * noise_scale
        )
        candidate_ox_mask = (
            optimization_targets[4].detach()
            - lr_atom * optimization_targets[4].grad
            + torch.randn_like(optimization_targets[4]) * noise_scale
        )
    else:
        candidate_atomic_distribution = optimization_targets[3].detach()
        candidate_ox_mask = optimization_targets[4].detach()

    # 5. 候補更新値をパラメータに反映（in-place更新）
    optimization_targets[0].data.copy_(candidate_scaled_batch_abc)
    optimization_targets[1].data.copy_(candidate_scaled_batch_angle)
    optimization_targets[2].data.copy_(candidate_batch_dir_coords)
    optimization_targets[3].data.copy_(candidate_atomic_distribution)
    optimization_targets[4].data.copy_(candidate_ox_mask)

    # 6. 更新後の候補状態で再評価（torch.no_grad()を用いる）
    with torch.no_grad():
        # forward_propagation_Crystalformer を呼び出して候補状態での出力を得る
        bandgap_pred_c, ef_pred_c, output_dict_c = forward_propagation_Crystalformer(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=temp,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=None,
            limit_coords_displacement=limit_coords_displacement,
            angle_range=angle_range,
            abc_range=abc_range,
        )
        # calculate_loss_from_output を呼び出し、候補状態での各サンプル損失を得る
        # 戻り値は (total_loss_candidate, candidate_each_loss, gap_loss, ef_loss, tolerance_loss, tolerance)
        _, candidate_each_loss, _, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_pred_c,
            ef_pred=ef_pred_c,
            sharpened_ox_mask=output_dict_c["sharpened_ox_mask"],
            normalized_dist=output_dict_c["normalized_dist"],
            site_ids=mini_batch_inputs_dict["site_ids"].to("cuda"),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_value,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

    # 7. SAの受け入れ判定：各サンプルごとに候補損失と元の損失を比較
    delta_loss = candidate_each_loss - each_loss  # shape: [batch_size]
    accept_prob = torch.exp(-delta_loss / temp)
    rand_vals = torch.rand_like(delta_loss)
    accept_mask = (delta_loss < 0) | (rand_vals < accept_prob)  # Trueなら候補更新を採用

    # 8. 受け入れなかったサンプルは保存しておいた旧パラメータに戻す
    def revert_param(param, old_param, mask):
        mask_expanded = mask.view(-1, *([1] * (param.dim() - 1)))
        new_val = torch.where(mask_expanded, param, old_param)
        param.data.copy_(new_val)

    revert_param(optimization_targets[0], old_params[0], accept_mask)
    revert_param(optimization_targets[1], old_params[1], accept_mask)
    revert_param(
        optimization_targets[2],
        old_params[2],
        expand_accept_mask(accept_mask, cry_atom_data).to(accept_mask.device),
    )
    revert_param(
        optimization_targets[3],
        old_params[3],
        expand_accept_mask(accept_mask, cry_atom_data).to(accept_mask.device),
    )
    revert_param(optimization_targets[4], old_params[4], accept_mask)

    # 9. 勾配リセット（次回ステップに備える）
    for param in optimization_targets:
        if param.grad is not None:
            param.grad.zero_()

    # 10. 更新後の各サンプル損失は、受け入れたサンプルは候補損失、そうでなければ元のeach_lossを採用し、その平均を算出
    updated_loss = (
        torch.where(accept_mask, candidate_each_loss, each_loss).mean().item()
    )
    additional_info["accept_mask"] = accept_mask
    additional_info["mode"] = "sa"

    return updated_loss, additional_info


# In[ ]:


import math
from typing import Any, Dict, List, Tuple

import torch


# Adam
def simulated_annearling_update(
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[torch.optim.lr_scheduler._LRScheduler],
    optimization_targets: list,
    step_i: int,
    total_loss: torch.Tensor,
    each_loss: torch.Tensor,
    temp: float,
    learning_rates: List[float],  # [lr_lattice, lr_atom, lr_coords]
    sa_noise_scale: float,
    step_of_x_opt_stop: int,
    angle_optimization: bool,
    cry_atom_data: torch.Tensor,
    mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    limit_coords_displacement: float,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    Ef_coef_value: float,
    crystal_system: torch.nn.Module,
    e_form_min: float,
    radii_tensor: torch.Tensor,
    abc_range: Tuple[float],
    angle_range: Tuple[float],
) -> Tuple[float, Dict[str, Any]]:
    """
    パラメータ別にAdamで更新幅を計算して、Simulated Annealing 受け入れ判定するバージョン
    """

    """
    改良版：外部で作成したOptimizerを受け取り、
    パラメータ操作は torch.no_grad() によって安全に実行します。
    """
    additional_info: Dict[str, Any] = {}

    print(f"step {step_i} tenperature: ", temp)

    # 1. 勾配リセット
    for opt in optimizers:
        opt.zero_grad()

    # 2. 勾配計算
    total_loss.backward()

    # 3. 古いパラメータを保存
    old_params = [param.detach().clone() for param in optimization_targets]

    # 4. 仮の更新を実行（momentum, Adamの内部状態も保持されたまま step()）
    for opt in optimizers:
        opt.step()
    # 4.1. scheduler.step() を実行
    for scheduler in schedulers:
        scheduler.step()

    # 5. delta を計算
    deltas = [
        param.detach() - old for param, old in zip(optimization_targets, old_params)
    ]

    # 6. 元のパラメータに戻す
    with torch.no_grad():
        for param, old in zip(optimization_targets, old_params):
            param.copy_(old)

    # 7. SA 用ノイズ付き候補を生成
    noise_scale = sa_noise_scale * temp
    candidates = [
        old + delta + torch.randn_like(delta) * noise_scale
        for old, delta in zip(old_params, deltas)
    ]

    # 8. 候補パラメータを一時適用
    with torch.no_grad():
        for param, cand in zip(optimization_targets, candidates):
            param.copy_(cand)

    # 9. 候補状態で損失を評価
    with torch.no_grad():
        bandgap_pred_c, ef_pred_c, output_dict_c = forward_propagation_Crystalformer(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=temp,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=None,
            limit_coords_displacement=limit_coords_displacement,
            angle_range=angle_range,
            abc_range=abc_range,
        )
        _, candidate_each_loss, _, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_pred_c,
            ef_pred=ef_pred_c,
            sharpened_ox_mask=output_dict_c["sharpened_ox_mask"],
            normalized_dist=output_dict_c["normalized_dist"],
            site_ids=mini_batch_inputs_dict["site_ids"].to(bandgap_pred_c.device),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_value,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

    # 10. SA の受け入れ判定
    delta_loss = candidate_each_loss - each_loss
    accept_prob = torch.exp(-delta_loss / temp)
    rand_vals = torch.rand_like(delta_loss)
    accept_mask = (delta_loss < 0) | (rand_vals < accept_prob)

    # 11. 拒否したサンプルのみ古いパラメータに戻す
    with torch.no_grad():
        for idx, (param, old, cand) in enumerate(
            zip(optimization_targets, old_params, candidates)
        ):
            # 座標系などサンプル次元ごとにマスクが必要なものは expand_accept_mask を使う
            if idx in (2, 3):
                mask = expand_accept_mask(accept_mask, cry_atom_data).to(old.device)
            else:
                mask = accept_mask
            mask_expanded = mask.view(-1, *([1] * (param.dim() - 1)))
            param.copy_(torch.where(mask_expanded, cand, old))

    # 12. 最終的な勾配をクリア
    for opt in optimizers:
        opt.zero_grad()

    # 13. 更新後の loss を返却
    updated_loss = (
        torch.where(accept_mask, candidate_each_loss, each_loss).mean().item()
    )

    additional_info["accept_mask"] = accept_mask
    additional_info["mode"] = "sa_adam_per_param"

    return updated_loss, additional_info


# In[ ]:


from typing import Any, Dict, List, Tuple

import torch


def simulated_annealing_noise_update(
    optimization_targets: List[torch.nn.Parameter],
    optimization_target_names: List[str],
    temp: float,
    sa_noise_scale: float,
    cry_atom_data: torch.Tensor,
    mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    limit_coords_displacement: float,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    Ef_coef_value: float,
    crystal_system: torch.nn.Module,
    e_form_min: float,
    radii_tensor: torch.Tensor,
    abc_range: Tuple[float, float, float],
    angle_range: Tuple[float, float, float],
) -> Tuple[float, Dict[str, Any]]:
    """
    Simulated Annealing noise update:
      1) 現在のパラメータで損失 orig_loss を計算
      2) ノイズを付与して候補パラメータへ
      3) 候補パラメータで損失 cand_loss を計算
      4) Δ = cand_loss - orig_loss で受容判定
      5) 受容しないサンプルは元に戻す
    """

    additional_info: Dict[str, Any] = {}
    # 温度ゼロ対策
    temp = max(temp, 1e-8)

    # —— 1) 現在の損失をまず計算 —— #
    with torch.no_grad():
        # Crystalformer の順伝播
        bandgap_pred_o, ef_pred_o, output_dict_o = forward_propagation_Crystalformer(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=temp,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=None,
            limit_coords_displacement=limit_coords_displacement,
            angle_range=angle_range,
            abc_range=abc_range,
        )
        # 損失計算
        _, orig_each_loss, _, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_pred_o,
            ef_pred=ef_pred_o,
            sharpened_ox_mask=output_dict_o["sharpened_ox_mask"],
            normalized_dist=output_dict_o["normalized_dist"],
            site_ids=mini_batch_inputs_dict["site_ids"].to(bandgap_pred_o.device),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_value,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

    # —— 2) ノイズ付与 —— #
    # 勾配更新後のパラメータをコピー
    updated_params = [param.detach().clone() for param in optimization_targets]
    # ノイズスケールは温度に比例
    noise_scale = sa_noise_scale * math.sqrt(temp)  # √T 依存にしてみるのもお勧め
    candidates = [upd + torch.randn_like(upd) * noise_scale for upd in updated_params]
    # 候補パラメータを適用
    with torch.no_grad():
        for p, cand in zip(optimization_targets, candidates):
            p.copy_(cand)

    # —— 3) ノイズ後の損失を計算 —— #
    with torch.no_grad():
        bandgap_pred_c, ef_pred_c, output_dict_c = forward_propagation_Crystalformer(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=temp,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=None,
            limit_coords_displacement=limit_coords_displacement,
            angle_range=angle_range,
            abc_range=abc_range,
        )
        _, cand_each_loss, _, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_pred_c,
            ef_pred=ef_pred_c,
            sharpened_ox_mask=output_dict_c["sharpened_ox_mask"],
            normalized_dist=output_dict_c["normalized_dist"],
            site_ids=mini_batch_inputs_dict["site_ids"].to(bandgap_pred_c.device),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_value,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

    # —— 4) 受容判定 —— #

    delta = cand_each_loss - orig_each_loss  # Δloss
    accept_prob = torch.exp(-delta / temp)
    rand_vals = torch.rand_like(delta, device=delta.device)  # 同じデバイスにする
    accept_mask = (delta < 0) | (rand_vals < accept_prob)

    # —— 5) 拒否サンプルは元に戻す —— #
    with torch.no_grad():
        for idx, (p_name, param, orig, cand) in enumerate(
            zip(
                optimization_target_names,
                optimization_targets,
                updated_params,
                candidates,
            )
        ):
            # 位置パラメータだけ特殊マスク展開が必要ならここで
            if idx in (2, 3):
                # サンプル次元ごとにマスクを展開
                mask = expand_accept_mask(accept_mask, cry_atom_data).to(param.device)
            else:
                # それ以外はそのまま
                mask = accept_mask
            # マスクをパラメータの次元に合わせて展開

            mask = mask.view(-1, *([1] * (param.dim() - 1)))
            param.copy_(torch.where(mask, cand, orig))

    # バッチ平均したスカラー損失を返す
    updated_loss = (
        torch.where(accept_mask, cand_each_loss, orig_each_loss).mean().item()
    )

    additional_info.update(
        {
            "orig_each_loss": orig_each_loss,
            "cand_each_loss": cand_each_loss,
            "delta_loss": delta,
            "accept_mask": accept_mask,
            "noise_scale": noise_scale,
        }
    )

    return updated_loss, additional_info


# In[ ]:


def parse_evolution_parameters(optimize_mode_str: str) -> dict:
    """Parses the optimize_mode string to extract parameters for the evolutionary update.

    The optimize_mode string is expected to contain the evolution parameters after a
    'pytorch_optimizer___' delimiter. Individual settings are separated by '__', and each
    setting consists of a key and one or more values separated by '-' (the first '-' is used
    as the separator). If a parameter not defined in the default list is encountered, a
    ValueError is raised.

    Default parameters:
        top_ratio: float = 0.25,
        mutation_noise: float = 0.05,
        steps_cm: List[int] = [50, 100, 150]

    For example, the input string:
        "pytorch_optimizer___steps_cm-50-100-150__mutation_noise-0.05"
    will yield a dictionary:
        {
            "top_ratio": 0.25,            # default remains because not specified
            "mutation_noise": 0.05,
            "steps_cm": [50, 100, 150]
            "group_C_use_rate": 0.1,
            "atom_dist_init": True,
        }

    Also, if the input is:
        "pytorch_optimizer___steps_cm-10"
    then the resulting dictionary will have:
        "steps_cm": [10]

    Args:
        optimize_mode_str (str): The optimization mode string.

    Returns:
        dict: A dictionary containing the evolution parameters.

    Raises:
        ValueError: If an unknown parameter is found in the input string.
    """
    # Default parameters for evolutionary update
    defaults = {
        "top_ratio": 0.25,
        "mutation_noise": 0.05,
        "steps_cm": [50, 100, 150],
        "group_C_use_rate": 0.1,
        "atom_dist_init": True,
    }

    # Check if the string starts with "pytorch_optimizer"
    if optimize_mode_str.startswith("pytorch_optimizer"):
        parts = optimize_mode_str.split("___", 1)
        if len(parts) == 2:
            param_str = parts[1]
        else:
            return defaults
    else:
        # If not starting with "pytorch_optimizer", assume no evolution parameters provided.
        return defaults

    # Split the parameters using '__' as delimiter.
    settings = param_str.split("__")
    for setting in settings:
        # Each setting is expected to be in the format "key-value" or "key-value1-value2-..."
        tokens = setting.split("-")
        if len(tokens) < 2:
            continue  # skip empty or malformed setting
        key = tokens[0].strip()
        # Check if the key is valid.
        if key not in defaults:
            raise ValueError(f"Unknown evolution parameter: '{key}'")
        # Determine whether the default value is a list.
        if isinstance(defaults[key], list):
            # Always treat the value as a list, even if a single token is provided.
            values = []
            for token in tokens[1:]:
                token = token.strip()
                # Check if the token represents a number.
                if token.replace(".", "", 1).isdigit():
                    # Convert to float if it contains a dot, else int.
                    if "." in token:
                        values.append(float(token))
                    else:
                        values.append(int(token))
                else:
                    values.append(token)
            defaults[key] = values
        else:
            # For non-list defaults, use the first token after the key.
            value = tokens[1].strip()
            if value.replace(".", "", 1).isdigit():
                value_converted = float(value) if "." in value else int(value)
            else:
                value_converted = value
            defaults[key] = value_converted

    return defaults


# In[ ]:


# In[ ]:


import random
from typing import List, Optional

import torch


def copy_and_mutate_update(
    optimization_targets: List[torch.Tensor],
    optimization_target_names: List[str],
    current_learning_rates: List[float],
    crystal_system: Optional[str],
    cry_atom_data: torch.Tensor,
    size: torch.Tensor,
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    site_ids: torch.Tensor,
    radii_tensor: torch.Tensor,
    init_coords: torch.Tensor,
    each_loss: torch.Tensor,
    gap_loss: torch.Tensor,
    ef_loss: torch.Tensor,
    tolerance_loss: torch.Tensor,
    top_ratio: float,
    abc_scaler,  # 型は ABC_Scaler
    mutation_noise: float,
    group_C_use_rate: float,
    atom_dist_init: bool,
    angle_optimization: bool,
    current_step: int,
    total_steps: int,
    Ef_criteria: float = -0.5,
    tolerance_criteria: float = 0.1,
    test_mode: bool = False,
):
    """Performs an evolutionary update (copy and mutate) on mini-batch samples,
    and returns the updated parameters along with new cry_atom_data and new_size.

    For each sample in the mini-batch (assumed along dimension 0), the update is performed as follows:
      - Group A: Samples with gap_loss==0 and ef_loss<=Ef_criteria (and for 'perovskite', tolerance_loss<=tolerance_criteria)
                 are kept unchanged.
      - Group B: Among the remaining samples, the top fraction (specified by top_ratio) with the lowest each_loss
                 are kept unchanged.
      - Group C: For the remaining samples, each sample is replaced by a copy of a randomly chosen sample from
                 Group A (or Group A∪Group B in non-test mode) with added noise (scaled by mutation_noise).

    For parameters whose shape can vary (i.e. "batch_dir_coords" and "atomic_distribution"),
    the global parameter tensor (shape [total_atoms, D]) is updated as follows:
      - Extract indices corresponding to the target sample (selected_indices) from cry_atom_data.
      - Split the global tensor into three parts: front, target block, and back.
      - For the chosen good sample, extract its block (good_indices), adjust the number of rows to match the target's
        atom count, add mutation noise to create mutated_block.
      - Reconstruct the global tensor by concatenating: front, mutated_block, back.
      - Also update cry_atom_data for the target block: assign the chosen sample's crystal ID.

    Finally, new_size is computed as a tensor of shape (original_batch_size,) where each element is the count of
    atoms assigned to that crystal ID in new_cry_atom_data.

    Args:
        optimization_targets (List[torch.Tensor]): List of parameter tensors. [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal]
        optimization_target_names (List[str]): List of names corresponding to the parameters. ['scaled_batch_abc', 'scaled_batch_angle', 'batch_dir_coords', 'atomic_distribution', 'ox_mask_learnable_tensor_per_crystal']
        current_learning_rates (List[float]): Learning rates for each group.
        crystal_system (Optional[str]): e.g. 'perovskite'; affects Group A criteria.
        cry_atom_data (torch.Tensor): 1D tensor mapping each row (atom) to a sample (crystal) ID.
        each_loss (torch.Tensor): 1D tensor of per-sample loss (shape: [batch_size]).
        gap_loss (torch.Tensor): 1D tensor of per-sample gap loss.
        ef_loss (torch.Tensor): 1D tensor of per-sample ef loss.
        tolerance_loss (torch.Tensor): 1D tensor of per-sample tolerance loss.
        top_ratio (float): Fraction of non-Group-A samples to designate as Group B.
        abc_scaler: An instance of ABC_Scaler (for updating batch_abc if needed).
        mutation_noise (float): Noise scale for mutation.
        group_C_use_rate (float): Probability threshold for skipping update in Group C.
        atom_dist_init (bool): Flag related to atomic distribution initialization.
        Ef_criteria (float, optional): Threshold for ef_loss for Group A. Defaults to -0.5.
        tolerance_criteria (float, optional): Threshold for tolerance_loss (for perovskite). Defaults to 0.1.
        test_mode (bool, optional): If True, use only Group A for mutation and set mutation_noise to 0. Defaults to True.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
            - Updated list of parameter tensors.
            - new_cry_atom_data: Updated cry_atom_data tensor.
            - new_size: 1D tensor (length = original batch_size) where each element is the count of atoms for that crystal.
    """
    batch_size = each_loss.size(0)
    assert batch_size <= cry_atom_data.size(
        0
    ), "batch_size must be <= cry_atom_data size"

    # Identify Group A:
    if crystal_system == "perovskite":
        group_A = (
            (gap_loss == 0)
            & (ef_loss <= Ef_criteria)
            & (tolerance_loss <= tolerance_criteria)
        ).nonzero(as_tuple=True)[0]
    else:
        group_A = ((gap_loss == 0) & (ef_loss <= Ef_criteria)).nonzero(as_tuple=True)[0]

    # Identify non-A indices (Group B and C)
    if crystal_system == "perovskite":
        non_A_mask = ~(
            (gap_loss == 0)
            & (ef_loss <= Ef_criteria)
            & (tolerance_loss <= tolerance_criteria)
        )
    else:
        non_A_mask = ~((gap_loss == 0) & (ef_loss <= Ef_criteria))

    non_A_indices = non_A_mask.nonzero(as_tuple=True)[0]

    # Among non-A, select top_ratio fraction with lowest each_loss as Group B.
    if non_A_indices.numel() > 0:
        non_A_losses = each_loss[non_A_indices]
        num_B = max(1, int(top_ratio * non_A_indices.numel()))
        _, sorted_indices = torch.sort(non_A_losses)
        group_B = non_A_indices[sorted_indices[:num_B]]
    else:
        group_B = torch.tensor([], dtype=torch.long, device=each_loss.device)

    # Group C: Non-A samples that are not in Group B.
    group_B_set = set(group_B.tolist())
    group_C = [i for i in non_A_indices.tolist() if i not in group_B_set]

    if test_mode:
        print(f"<info> test mode in copy_and_mutate_update")

        print(type(group_A), group_A)
        print(type(group_B), group_B)
        print(type(group_C), group_C)

        group_C = set(group_B.tolist() + group_C)
        group_B = torch.tensor([])
        print(
            f"Group A: {group_A.tolist()}, Group B: {group_B.tolist()}, Group C: {group_C}"
        )
        print(f"Test choice group (A and B) : {(group_A.tolist() + group_B.tolist())}")
        mutation_noise = 0.0
        group_C_use_rate = -1.0
        atom_dist_init = False

    # new_cry_atom_data: Initialize with the original cry_atom_data.
    new_cry_atom_data = cry_atom_data.clone()
    new_size = size.clone()

    # For each sample i in Group C, perform update.
    with torch.no_grad():
        for i in group_C:
            # Randomly decide whether to skip the update for this sample.
            if random.random() < group_C_use_rate:
                continue
            # Choose a good crystal from Group A or Group B.
            chosen_good_crystal = int(
                random.choice((group_A.tolist() + group_B.tolist()))
            )

            # サンプル i の更新開始前に、インデックスを事前計算して保存する
            orig_selected_indices = (new_cry_atom_data == i).nonzero(as_tuple=True)[0]
            orig_good_indices = (new_cry_atom_data == chosen_good_crystal).nonzero(
                as_tuple=True
            )[0]

            # Update each parameter in optimization_targets.
            for param, param_name in zip(
                optimization_targets, optimization_target_names
            ):
                if param_name in [
                    "scaled_batch_abc",
                    "scaled_batch_angle",
                    "ox_mask_learnable_tensor_per_crystal",
                ]:
                    # Update fixed parameters (assumed shape [batch_size, ...])
                    if (not angle_optimization) and param_name == "scaled_batch_angle":
                        pass
                    elif param_name == "scaled_batch_abc":
                        noise = (
                            torch.randn_like(param[chosen_good_crystal])
                            * mutation_noise
                        )
                        param[i].copy_(param[chosen_good_crystal].clone() + noise)
                        if test_mode:
                            print(f"i: {i}, chosen_good_crystal: {chosen_good_crystal}")
                            print(
                                f"(Before) abc_scaler.init_batch_abc[i]: {abc_scaler.init_batch_abc[i]}"
                            )
                        abc_scaler.init_batch_abc[i] = abc_scaler.init_batch_abc[
                            chosen_good_crystal
                        ]
                        # あとで消す
                        if test_mode:
                            print(
                                f"(After) abc_scaler.init_batch_abc[i]: {abc_scaler.init_batch_abc[i]}"
                            )
                            print(
                                f"abc_scaler.init_batch_abc[chosen_good_crystal]: {abc_scaler.init_batch_abc[chosen_good_crystal]}"
                            )

                    else:
                        noise = (
                            torch.randn_like(param[chosen_good_crystal])
                            * mutation_noise
                        )
                        param[i].copy_(param[chosen_good_crystal].clone() + noise)
                else:  # param_name in ['batch_dir_coords', 'atomic_distribution']:
                    # For variable shape parameters:
                    # param shape: [total_atoms, D]
                    # Get indices for sample i and for chosen_good_crystal.
                    start_i = orig_selected_indices[0].item()
                    end_i = orig_selected_indices[-1].item() + 1
                    front_tensor = param[:start_i]
                    back_tensor = param[end_i:]

                    # If the number of atoms differs, adjust the size of good_block.
                    if param_name == "atomic_distribution" and atom_dist_init:
                        # re-initialize atomic_distribution using uniform distribution
                        good_block = (
                            torch.ones_like(param[orig_good_indices].clone()) * i
                        )
                    else:
                        # Determine the target count for sample i.
                        good_block = param[orig_good_indices].clone()

                    mutated_block = (
                        good_block + torch.randn_like(good_block) * mutation_noise
                    )
                    new_param = torch.cat(
                        [front_tensor, mutated_block, back_tensor], dim=0
                    )
                    param.data = new_param.clone()

                    if param_name == "atomic_distribution":
                        # update atomic_mask and ox_states_used_mask
                        ## atomic_mask update:
                        front_tensor = atomic_mask[:start_i]
                        back_tensor = atomic_mask[end_i:]
                        good_block = atomic_mask[orig_good_indices].clone()
                        atomic_mask = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )
                        ## ox_states_used_mask update:
                        front_tensor = ox_states_used_mask[:start_i]
                        back_tensor = ox_states_used_mask[end_i:]
                        good_block = ox_states_used_mask[orig_good_indices].clone()
                        ox_states_used_mask = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # update init_coords
                        front_tensor = init_coords[:start_i]
                        back_tensor = init_coords[end_i:]
                        good_block = init_coords[orig_good_indices].clone()
                        init_coords = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # Update site_ids for the target block.
                        front_tensor = site_ids[:start_i]
                        back_tensor = site_ids[end_i:]
                        good_block = site_ids[orig_good_indices].clone()
                        site_ids = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # Update radii_tensor fofr the target block.
                        front_tensor = radii_tensor[:start_i]
                        back_tensor = radii_tensor[end_i:]
                        good_block = radii_tensor[orig_good_indices].clone()
                        radii_tensor = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # Update cry_atom_data for the target block.
                        orig_selected_indices = (new_cry_atom_data == i).nonzero(
                            as_tuple=True
                        )[0]
                        orig_good_indices = (
                            new_cry_atom_data == chosen_good_crystal
                        ).nonzero(as_tuple=True)[0]
                        start_i = orig_selected_indices[0].item()
                        end_i = orig_selected_indices[-1].item() + 1
                        # Update cry_atom_data for the target block.
                        # cry_atom_data update:
                        front_tensor = new_cry_atom_data[:start_i]
                        back_tensor = new_cry_atom_data[end_i:]
                        mutated_block = torch.ones_like(good_block) * i
                        mutated_block = torch.full(
                            (good_block.shape[0],),
                            i,
                            dtype=cry_atom_data.dtype,
                            device=cry_atom_data.device,
                        )
                        new_cry_atom_data = torch.cat(
                            [front_tensor, mutated_block, back_tensor], dim=0
                        )
                        # size update:
                        new_size[i] = mutated_block.numel()

    # Update the parameters in optimization_targets with the new values.
    new_optimization_targets = []
    with torch.no_grad():
        for param in optimization_targets:
            # params to learnable parameters
            new_optimization_targets.append(
                torch.nn.Parameter(param.data.clone().detach(), requires_grad=True)
            )

    # Update abc_scaler base length.
    abc_scaler.init_batch_abc
    abc_scaler.update_base_length()

    # optimizer and scheduler
    (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    ) = define_optimizer_and_scheduler(
        lattice_lr=current_learning_rates[0],
        atom_lr=current_learning_rates[1],
        coords_lr=current_learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=new_optimization_targets[
            4
        ],  # ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=new_optimization_targets[3],  # atomic_distribution,
        scaled_batch_abc=new_optimization_targets[0],  # scaled_batch_abc,
        scaled_batch_angle=new_optimization_targets[1],  # scaled_batch_angle,
        batch_dir_coords=new_optimization_targets[2],  # batch_dir_coords,
        angle_optimization=angle_optimization,
        num_steps=total_steps - current_step,
        lattice_cycle=1,
        atom_cycle=1,
        coords_cycle=1,
    )
    optimizers_and_schedulers = (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    )

    return (
        new_optimization_targets,
        optimizers_and_schedulers,
        new_cry_atom_data,
        new_size,
        atomic_mask,
        ox_states_used_mask,
        site_ids,
        radii_tensor,
        init_coords,
        abc_scaler,
    )


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


def update_parameters_step(
    optimization_targets: List[torch.Tensor],
    optimization_target_names: List[str],
    step_i: int,
    total_steps: int,
    total_loss: torch.Tensor,
    each_loss: torch.Tensor,
    gap_loss: torch.Tensor,
    ef_loss: torch.Tensor,
    tolerance_loss: torch.Tensor,
    learning_rates: List[float],  # [lr_lattice, lr_atom, lr_coords]
    step_of_x_opt_stop: int,
    optimize_mode: str,
    cry_atom_data: torch.Tensor,
    size: torch.Tensor,
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    site_ids: torch.Tensor,
    radii_tensor: torch.Tensor,
    init_coords: torch.Tensor,
    angle_optimization: bool,
    optimizer_lattice: torch.optim.Optimizer,
    optimizer_atom: torch.optim.Optimizer,
    optimizer_coords: torch.optim.Optimizer,
    scheduler_lattice: torch.optim.lr_scheduler._LRScheduler,
    scheduler_atom: torch.optim.lr_scheduler._LRScheduler,
    scheduler_coords: torch.optim.lr_scheduler._LRScheduler,
    fixed_inputs: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    limit_coords_displacement: float,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: Any,
    Ef_coef_value: float,
    crystal_system: Optional[str],
    e_form_min: float,
    abc_range: Tuple[float, float],
    angle_range: Tuple[float, float],
) -> Tuple[float, Dict[str, Any]]:
    """
    1ステップ分のパラメータ更新を行います。
    optimize_mode が "adam" の場合は通常のAdam更新、"sa" の場合はSA更新を実行します。

    Returns:
        updated_loss (float): 更新後の平均損失
        additional_info (Dict[str, Any]): 追加情報（SAの場合は受け入れマスクなど）
    """
    additional_info = {}
    optimizer_lattice.zero_grad()
    optimizer_atom.zero_grad()
    optimizer_coords.zero_grad()

    total_loss.backward()

    if step_i < step_of_x_opt_stop:
        optimizer_lattice.step()
        optimizer_atom.step()
        optimizer_coords.step()
    else:
        optimizer_lattice.step()
        optimizer_coords.step()

    scheduler_lattice.step()
    scheduler_atom.step()
    scheduler_coords.step()

    updated_loss = total_loss.item()
    additional_info["mode"] = "adam"
    additional_info["optimizer_lattice"] = optimizer_lattice
    additional_info["optimizer_atom"] = optimizer_atom
    additional_info["optimizer_coords"] = optimizer_coords
    additional_info["scheduler_lattice"] = scheduler_lattice
    additional_info["scheduler_atom"] = scheduler_atom
    additional_info["scheduler_coords"] = scheduler_coords
    additional_info["new_cry_atom_data"] = cry_atom_data
    additional_info["new_size"] = size
    additional_info["atomic_mask"] = atomic_mask
    additional_info["ox_states_used_mask"] = ox_states_used_mask
    additional_info["abc_scaler"] = scalers["abc_scaler"]
    additional_info["site_ids"] = site_ids
    additional_info["radii_tensor"] = radii_tensor
    additional_info["init_coords"] = fixed_inputs["init_coords"]

    # get current learning rates from scheduler_coords, optimizer_lattice, optimizer_atom
    current_learning_rates = [
        scheduler_lattice.get_last_lr()[0],
        optimizer_atom.param_groups[0]["lr"],
        optimizer_coords.param_groups[0]["lr"],
    ]

    # simulated annearling

    if "sa___" in optimize_mode:
        if not len(optimize_mode.split("sa___")) == 2:
            print(f"optimize_mode: {optimize_mode}")
            print(optimize_mode.split("sa___"))
            raise ValueError(
                f"optimize_mode should be like sa___<mode>_<parameters>, but got {optimize_mode}"
            )

        sa_mode = optimize_mode.split("sa___")[1]
        sa_parmas_dict = parse_sa_parameters("sa___" + sa_mode)
        step_i_temp4sa = get_current_temperature(
            step_i=step_i,
            total_steps=total_steps,
            schedule_type="exponential",
            T0=sa_parmas_dict["T0"],
            T_min=sa_parmas_dict["T_min"],
            alpha=sa_parmas_dict["alpha"],
            beta=sa_parmas_dict["beta"],
            c=sa_parmas_dict["c"],
        )
        simulated_annealing_noise_update(
            optimization_targets=optimization_targets,
            optimization_target_names=optimization_target_names,
            temp=step_i_temp4sa,
            sa_noise_scale=sa_parmas_dict["noise_scale"],
            cry_atom_data=cry_atom_data,
            mini_batch_inputs_dict=fixed_inputs,
            scalers=scalers,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            limit_coords_displacement=limit_coords_displacement,
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            Ef_coef_value=Ef_coef_value,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
            abc_range=abc_range,
            angle_range=angle_range,
        )

    # Perform evolutionary update after the optimizer step:
    # Copy parameters from selected samples (Group B) to others (Group C) and add mutation noise.
    parse_evolution_parameters_dict = parse_evolution_parameters(
        optimize_mode.split("sa___")[0]
    )
    if step_i in parse_evolution_parameters_dict["steps_cm"]:
        print(
            "---------------------------- \n    <info> Performing copy and mutate update... \n ----------------------------"
        )
        (
            optimization_targets,
            optimizers_and_schedulers,
            new_cry_atom_data,
            new_size,
            atomic_mask,
            ox_states_used_mask,
            site_ids,
            radii_tensor,
            init_coords,
            abc_scaler,
        ) = copy_and_mutate_update(
            optimization_targets=optimization_targets,
            optimization_target_names=optimization_target_names,
            crystal_system=crystal_system,
            cry_atom_data=cry_atom_data,
            size=size,
            atomic_mask=atomic_mask,
            ox_states_used_mask=ox_states_used_mask,
            site_ids=site_ids,
            radii_tensor=radii_tensor,
            init_coords=init_coords,
            current_learning_rates=current_learning_rates,
            current_step=step_i,
            total_steps=total_steps,
            angle_optimization=angle_optimization,
            each_loss=each_loss,
            gap_loss=gap_loss,
            ef_loss=ef_loss,
            tolerance_loss=tolerance_loss,
            abc_scaler=scalers["abc_scaler"],
            group_C_use_rate=parse_evolution_parameters_dict["group_C_use_rate"],
            atom_dist_init=parse_evolution_parameters_dict["atom_dist_init"],
            top_ratio=parse_evolution_parameters_dict["top_ratio"],
            mutation_noise=parse_evolution_parameters_dict["mutation_noise"],
        )
        # optimizers_and_schedulers
        additional_info["optimizer_lattice"] = optimizers_and_schedulers[0]
        additional_info["optimizer_atom"] = optimizers_and_schedulers[1]
        additional_info["optimizer_coords"] = optimizers_and_schedulers[2]
        additional_info["scheduler_lattice"] = optimizers_and_schedulers[3]
        additional_info["scheduler_atom"] = optimizers_and_schedulers[4]
        additional_info["scheduler_coords"] = optimizers_and_schedulers[5]
        # Update cry_atom_data and size
        additional_info["new_cry_atom_data"] = new_cry_atom_data
        additional_info["new_size"] = new_size
        additional_info["atomic_mask"] = atomic_mask
        additional_info["ox_states_used_mask"] = ox_states_used_mask
        additional_info["site_ids"] = site_ids
        additional_info["radii_tensor"] = radii_tensor
        additional_info["init_coords"] = init_coords

        # Update abc_scaler
        additional_info["abc_scaler"] = abc_scaler

        """
        # 更新しているわけではないが、optimizerとschedulerの情報を保持しておく
        additional_info["optimizer_lattice"] = optimizer_lattice
        additional_info["optimizer_atom"] = optimizer_atom
        additional_info["optimizer_coords"] = optimizer_coords
        additional_info["scheduler_lattice"] = scheduler_lattice
        additional_info["scheduler_atom"] = scheduler_atom
        additional_info["scheduler_coords"] = scheduler_coords
        additional_info["new_cry_atom_data"] = cry_atom_data
        additional_info["new_size"] = size
        additional_info["atomic_mask"] = atomic_mask
        additional_info["ox_states_used_mask"] = ox_states_used_mask
        additional_info["abc_scaler"] = scalers['abc_scaler']
        additional_info["site_ids"] = site_ids
        additional_info["radii_tensor"] = radii_tensor
        additional_info["init_coords"] = fixed_inputs['init_coords']
        """

    return updated_loss, optimization_targets, additional_info


# In[ ]:


# In[ ]:


# ### Crystalformerの最適化と評価関数

# In[ ]:


def Crystalformer_prediction(
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    normalized_dist: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    trans_vec: torch.Tensor,
    size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # train modeならばevalmodeにする
    if bandgap_model.training:
        bandgap_model.eval()
    if e_form_model.training:
        e_form_model.eval()
    bandgap_pred = torch.clip(
        bandgap_model((normalized_dist, pos, batch, trans_vec, size)), min=0
    )
    ef_pred = e_form_model((normalized_dist, pos, batch, trans_vec, size))

    return bandgap_pred, ef_pred


# In[ ]:


def forward_propagation_Crystalformer(
    optimization_targets: List[torch.Tensor],
    fixed_inputs: Dict[str, torch.Tensor],
    scalers: dict,
    temperature: float,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    adding_noise_scale: Optional[float],
    limit_coords_displacement: Optional[float],
    abc_range,
    angle_range,
):
    (
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ) = optimization_targets
    device = atomic_distribution.device
    size = fixed_inputs["size"].to(device)
    batch = fixed_inputs["batch"].to(device)
    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]
    batch_ori = fixed_inputs["batch"].to(device)
    atomic_mask = fixed_inputs["atomic_mask"].to(device)
    ox_states_used_mask = fixed_inputs["ox_states_used_mask"].to(device)
    init_coords = fixed_inputs["init_coords"].to(device)
    batch = batch_ori - batch_ori.min()  # batchを0から始まるように調整

    (
        batch_abc,
        batch_angle,
        normed_batch_dir_coords,
        normalized_dist,
        pos,
        lattice_vectors,
        sharpened_ox_mask,
    ) = Crystalformer_preprocess(
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        atomic_distribution=atomic_distribution,
        size=size,
        temperature=temperature,
        abc_scaler=abc_scaler,
        angle_scaler=angle_scaler,
        atomic_mask=atomic_mask,
        ox_states_used_mask=ox_states_used_mask,
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        init_coords=init_coords,
        limit_coords_displacement=limit_coords_displacement,
        adding_noise_scale=adding_noise_scale,
        abc_range=abc_range,
        angle_range=angle_range,
    )

    bandgap_pred, ef_pred = Crystalformer_prediction(
        bandgap_model=bandgap_model,
        e_form_model=e_form_model,
        normalized_dist=normalized_dist,
        pos=pos,
        batch=batch,
        trans_vec=lattice_vectors,
        size=size,
    )

    output_dict = {
        "bandgap_pred": bandgap_pred,
        "ef_pred": ef_pred,
        "batch_abc": batch_abc,
        "batch_angle": batch_angle,
        "lattice_vectors": lattice_vectors,
        "pos": pos,
        "normalized_dist": normalized_dist,
        "normed_batch_dir_coords": normed_batch_dir_coords,
        "batch_dir_coords": batch_dir_coords,
        "sharpened_ox_mask": sharpened_ox_mask,
        "batch": batch,
    }
    return bandgap_pred, ef_pred, output_dict


# In[ ]:


def optimize_solution_Crystalformer(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: dict,
    learning_rates: List[float],
    learning_rate_cycle: List[int],
    num_steps: int,
    dist_temp_sche: List[float],
    Ef_coef_schedule: List[float],
    step_of_x_opt_stop: int,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: Optional[torch.nn.Module],
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    scalers: dict,
    crystal_system: Optional[torch.nn.Module],
    limit_coords_displacement: Optional[float],
    e_form_min: Optional[float],
    adding_noise_scale: Optional[float],
    angle_optimization: bool = True,
    device: str = "cuda",
    time_series_save: bool = True,
):
    """
    Optimize the solution for a batch of crystal structures using Crystalformer.
    """

    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]
    atomic_distribution = torch.nn.Parameter(
        mini_batch_inputs_dict["atomic_distribution"].clone().to(device)
    )
    scaled_batch_abc = torch.nn.Parameter(
        abc_scaler.scale(mini_batch_inputs_dict["batch_abc"].clone().to(device))
    )
    scaled_batch_angle = torch.nn.Parameter(
        angle_scaler.scale(mini_batch_inputs_dict["batch_angle"].clone().to(device))
    )
    batch_dir_coords = torch.nn.Parameter(
        mini_batch_inputs_dict["batch_dir_coords"].clone().to(device)
    )
    ox_mask_learnable_tensor_per_crystal = torch.nn.Parameter(
        mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"]
        .clone()
        .to(device)
    )

    # Keep the template atom counts and lattice scaling in all four conditions.
    if settings_dict["initialization"].startswith("random-"):
        generator = settings_dict["initialization_generator"]
        scaled_batch_abc = torch.nn.Parameter(
            torch.rand(scaled_batch_abc.shape, device=device, generator=generator)
        )
        scaled_batch_angle = torch.nn.Parameter(
            torch.rand(scaled_batch_angle.shape, device=device, generator=generator)
        )
        batch_dir_coords = torch.nn.Parameter(
            torch.rand(batch_dir_coords.shape, device=device, generator=generator)
        )
    if settings_dict["initialization"].endswith("-random"):
        mini_batch_inputs_dict["atomic_mask"] = torch.ones_like(
            mini_batch_inputs_dict["atomic_mask"]
        )

    abc_range = settings_dict["abc_range"]
    angle_range = settings_dict["angle_range"]

    # optimizer and scheduler
    optimize_mode = settings_dict["optimize_mode"]
    if optimize_mode.startswith("pytorch_optimizer"):
        (
            optimizer_lattice,
            optimizer_atom,
            optimizer_coords,
            scheduler_lattice,
            scheduler_atom,
            scheduler_coords,
        ) = define_optimizer_and_scheduler(
            lattice_lr=learning_rates[0],
            atom_lr=learning_rates[1],
            coords_lr=learning_rates[2],
            ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
            atomic_distribution=atomic_distribution,
            scaled_batch_abc=scaled_batch_abc,
            scaled_batch_angle=scaled_batch_angle,
            batch_dir_coords=batch_dir_coords,
            angle_optimization=angle_optimization,
            num_steps=num_steps,
            lattice_cycle=learning_rate_cycle[0],
            atom_cycle=learning_rate_cycle[1],
            coords_cycle=learning_rate_cycle[2],
        )
    else:
        raise ValueError(f"Unknown optimize_mode: {optimize_mode}")

    # initialzation of history
    lattice_mat_history, coords_history, normed_coords_history, elememts_history = (
        [],
        [],
        [],
        [],
    )
    abc_history, angle_history = [], []
    gap_history, e_form_history, t_loss_history = [], [], []
    VALID_ELEMENTS98 = create_element_list()

    optimization_targets = [
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ]
    optimization_target_names = [
        "scaled_batch_abc",
        "scaled_batch_angle",
        "batch_dir_coords",
        "atomic_distribution",
        "ox_mask_learnable_tensor_per_crystal",
    ]

    for step_i in tqdm(range(num_steps), desc="step", leave=False):
        # ステップごとの処理

        bandgap_pred, ef_pred, output_dict = forward_propagation_Crystalformer(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=dist_temp_sche[step_i],
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            abc_range=abc_range,
            angle_range=angle_range,
        )
        total_loss, each_loss, gap_loss, ef_loss, tolerance_loss, tolerance = (
            calculate_loss_from_output(
                bandgap_pred=bandgap_pred,
                ef_pred=ef_pred,
                sharpened_ox_mask=output_dict["sharpened_ox_mask"],
                normalized_dist=output_dict["normalized_dist"],
                site_ids=mini_batch_inputs_dict["site_ids"].to(device),
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                ef_coef=Ef_coef_schedule[step_i],
                crystal_system=crystal_system,
                e_form_min=e_form_min,
                radii_tensor=mini_batch_inputs_dict["radii_tensor"],
            )
        )

        # Update parameters
        total_loss, optimization_targets, additional_info = update_parameters_step(
            optimization_targets=optimization_targets,
            optimization_target_names=optimization_target_names,
            step_i=step_i,
            total_loss=total_loss,
            gap_loss=gap_loss,
            ef_loss=ef_loss,
            tolerance_loss=tolerance_loss,
            each_loss=each_loss,
            total_steps=num_steps,
            learning_rates=learning_rates,
            cry_atom_data=output_dict["batch"].detach(),
            size=mini_batch_inputs_dict["size"].to(device),
            atomic_mask=mini_batch_inputs_dict["atomic_mask"],
            ox_states_used_mask=mini_batch_inputs_dict["ox_states_used_mask"],
            site_ids=mini_batch_inputs_dict["site_ids"].to(device),
            radii_tensor=mini_batch_inputs_dict["radii_tensor"].to(device),
            init_coords=mini_batch_inputs_dict["init_coords"].to(device),
            step_of_x_opt_stop=step_of_x_opt_stop,
            optimize_mode=optimize_mode,
            angle_optimization=angle_optimization,
            optimizer_lattice=optimizer_lattice,
            optimizer_atom=optimizer_atom,
            optimizer_coords=optimizer_coords,
            scheduler_lattice=scheduler_lattice,
            scheduler_atom=scheduler_atom,
            scheduler_coords=scheduler_coords,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            limit_coords_displacement=limit_coords_displacement,
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            Ef_coef_value=Ef_coef_schedule[step_i],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            abc_range=abc_range,
            angle_range=angle_range,
        )
        # update optimizer
        optimizer_atom = additional_info["optimizer_atom"]
        optimizer_lattice = additional_info["optimizer_lattice"]
        optimizer_coords = additional_info["optimizer_coords"]
        scheduler_lattice = additional_info["scheduler_lattice"]
        scheduler_atom = additional_info["scheduler_atom"]
        scheduler_coords = additional_info["scheduler_coords"]

        mini_batch_inputs_dict["batch"] = additional_info["new_cry_atom_data"].to(
            device
        )
        mini_batch_inputs_dict["size"] = additional_info["new_size"].to(device)
        mini_batch_inputs_dict["atomic_mask"] = additional_info["atomic_mask"].to(
            device
        )
        mini_batch_inputs_dict["ox_states_used_mask"] = additional_info[
            "ox_states_used_mask"
        ].to(device)
        mini_batch_inputs_dict["site_ids"] = additional_info["site_ids"].to(device)
        mini_batch_inputs_dict["radii_tensor"] = additional_info["radii_tensor"].to(
            device
        )
        mini_batch_inputs_dict["init_coords"] = additional_info["init_coords"].to(
            device
        )

        scalers["abc_scaler"] = additional_info["abc_scaler"]
        abc_scaler = additional_info["abc_scaler"]

        # 損失の記録
        gap_history.append(bandgap_pred.detach().cpu().numpy())
        e_form_history.append(ef_pred.detach().cpu().numpy())
        t_loss_history.append(tolerance.detach().cpu().numpy())

        if time_series_save:
            lattice_mat_history.append(
                output_dict["lattice_vectors"].detach().cpu().numpy()
            )
            normed_coords_history.append(
                output_dict["normed_batch_dir_coords"].detach().cpu().numpy()
            )
            coords_history.append(
                output_dict["batch_dir_coords"].detach().cpu().numpy()
            )
            abc_history.append(output_dict["batch_abc"].detach().cpu().numpy())
            angle_history.append(output_dict["batch_angle"].detach().cpu().numpy())
            elements = [
                VALID_ELEMENTS98[atom_idx]
                for atom_idx in torch.argmax(output_dict["normalized_dist"], dim=1)
            ]
            elememts_history.append(elements)

    # End of optimization
    (
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ) = optimization_targets
    print(
        f"The number of invalid crystral structures : {(torch.isnan(bandgap_pred)).detach().cpu().numpy().sum()}/{len(bandgap_pred)}"
    )
    # output_dict precedes the last update and may have a different atom count
    # after mutation. Evaluation recomputes all derived values from these final inputs.
    optimized_mini_batch_inputs_dict = mini_batch_inputs_dict.copy()
    optimized_mini_batch_inputs_dict["atomic_distribution"] = atomic_distribution
    optimized_mini_batch_inputs_dict["scaled_batch_abc"] = scaled_batch_abc
    optimized_mini_batch_inputs_dict["scaled_batch_angle"] = scaled_batch_angle
    optimized_mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
        ox_mask_learnable_tensor_per_crystal
    )
    optimized_mini_batch_inputs_dict["batch_dir_coords"] = batch_dir_coords
    optimized_mini_batch_inputs_dict["gap_history"] = np.transpose(
        np.array(gap_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["ef_history"] = np.transpose(
        np.array(e_form_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["t_history"] = np.transpose(
        np.array(t_loss_history).squeeze(), (1, 0)
    )
    time_series = [
        lattice_mat_history,
        coords_history,
        normed_coords_history,
        abc_history,
        angle_history,
        elememts_history,
    ]

    return (optimized_mini_batch_inputs_dict, time_series, scalers)


# In[ ]:


def evaluation_for_each_batch_Crystalformer(
    settings_dict: Dict[str, Any],
    optimized_mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    crystal_system: Optional[str],
    limit_coords_displacement: Optional[float],
    e_form_min: Optional[float],
    Ef_coef_schedule: List[float],
    adding_noise_scale: Optional[float],
    onehot_temperature: float = 1e-8,
):
    """
    Evaluate the optimized solution for a batch of crystal structures using Crystalformer.
    """
    scaled_batch_abc = optimized_mini_batch_inputs_dict["scaled_batch_abc"]
    scaled_batch_angle = optimized_mini_batch_inputs_dict["scaled_batch_angle"]
    batch_dir_coords = optimized_mini_batch_inputs_dict["batch_dir_coords"]
    atomic_distribution = optimized_mini_batch_inputs_dict["atomic_distribution"]
    device = atomic_distribution.device
    radii_tensor = optimized_mini_batch_inputs_dict["radii_tensor"]
    ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict[
        "ox_mask_learnable_tensor_per_crystal"
    ]
    size = optimized_mini_batch_inputs_dict["size"].to(device)
    batch_ori = optimized_mini_batch_inputs_dict["batch"].to(device)
    abc_range = settings_dict["abc_range"]
    angle_range = settings_dict["angle_range"]

    batch = batch_ori - batch_ori.min()  # batchを0から始まるように調整

    with torch.no_grad():
        # ステップごとの処理
        bandgap_dist, ef_dist, output_dict = forward_propagation_Crystalformer(
            optimization_targets=[
                scaled_batch_abc,
                scaled_batch_angle,
                batch_dir_coords,
                atomic_distribution,
                ox_mask_learnable_tensor_per_crystal,
            ],
            fixed_inputs=optimized_mini_batch_inputs_dict,
            scalers=scalers,
            temperature=onehot_temperature,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            angle_range=angle_range,
            abc_range=abc_range,
        )

        _, _, loss_dist, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_dist,
            ef_pred=ef_dist,
            sharpened_ox_mask=output_dict["sharpened_ox_mask"],
            normalized_dist=output_dict["normalized_dist"],
            site_ids=optimized_mini_batch_inputs_dict["site_ids"].to(device),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[-1],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

        # calculation with one-hot atomic distribution
        max_val, onehot_x, _, onehot_ox_mask = calculate_onehot(
            output_dict, atom_feat_matrix=None
        )
        bandgap_onehot, ef_onehot = Crystalformer_prediction(
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            normalized_dist=onehot_x,
            pos=output_dict["pos"],
            batch=batch,
            trans_vec=output_dict["lattice_vectors"],
            size=size,
        )

        _, _, loss_onehot, _, tolerance_loss, tolerance = calculate_loss_from_output(
            bandgap_pred=bandgap_onehot,
            ef_pred=ef_onehot,
            sharpened_ox_mask=onehot_ox_mask,
            normalized_dist=onehot_x,
            site_ids=optimized_mini_batch_inputs_dict["site_ids"].to(device),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[-1],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

    # 結果の記録
    optimized_mini_batch_inputs_dict = optimized_mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict["max_val"] = max_val
    optimized_mini_batch_inputs_dict["bandgap_onehot"] = bandgap_onehot
    optimized_mini_batch_inputs_dict["bandgap_dist"] = bandgap_dist
    optimized_mini_batch_inputs_dict["ef_onehot"] = ef_onehot
    optimized_mini_batch_inputs_dict["ef_dist"] = ef_dist
    optimized_mini_batch_inputs_dict["loss_onehot"] = loss_onehot
    optimized_mini_batch_inputs_dict["loss_dist"] = loss_dist
    optimized_mini_batch_inputs_dict["onehot_x"] = onehot_x
    optimized_mini_batch_inputs_dict["tolerance"] = tolerance
    optimized_mini_batch_inputs_dict["tolerance_loss"] = tolerance_loss
    optimized_mini_batch_inputs_dict["onehot_ox_mask"] = onehot_ox_mask

    return optimized_mini_batch_inputs_dict


# ### 両方を使った最適化と評価

# In[ ]:


def optimize_solution_Both(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: Dict[str, Any],
    learning_rates: List[float],
    learning_rate_cycle: List[int],
    num_steps: int,
    dist_temp_sche: List[float],
    Ef_coef_schedule: List[float],
    step_of_x_opt_stop: int,
    gap_loss_func: torch.nn.Module,
    bandgap_model: Dict[str, torch.nn.Module],
    e_form_model: Dict[str, torch.nn.Module],
    scalers: dict,
    adding_noise_scale: Optional[float],
    e_form_min: Optional[float],
    crystal_system: Optional[str],
    limit_coords_displacement: Optional[float],
    angle_optimization: bool = True,
    tolerance_loss_func=Optional[torch.nn.Module],
    device: str = "cuda",
    time_series_save: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    結晶の最適化を行う関数
    Args:
        mini_batch_inputs_dict: ALIGNNへの入力を構成するために必要な変数
            atomic_distribution(torch.Tensor): 原子分布, 学習可能パラメータ
            batch_abc(torch.Tensor): 結晶軸長さ, 学習可能パラメータ
            batch_angle(torch.Tensor): 結晶軸角度, 学習可能パラメータ
            num_edges(torch.Tensor): 結晶ごとのエッジ数, len(num_edges) == num_candidates
            batch_dir_coords(torch.Tensor): バッチ内部に存在する全原子の内部座標
            batch_dst_ids(torch.Tensor): バッチ内部に存在する全結合の終点のID
            batch_src_ids(torch.Tensor): バッチ内部に存在する全結合の始点のID
            batch_displace(torch.Tensor): バッチ内部に存在する全結合のdisplacement
            z_src_ids(torch.Tensor): ライングラフのエッジの始点のID
            z_dst_ids(torch.Tensor): ライングラフのエッジの終点のID
        num_steps(int): 最適化のステップ数
        dist_temp_sche(List[float]): Softmax温度スケジュール
        Ef_coef_schedule(List[float]): Ef損失の導入スケジュール
        step_of_x_opt_stop(int): xの最適化を止めるステップ数
        bandgap_model(torch.nn.Module): bandgapを予測するモデル
        e_form_model(torch.nn.Module): Efを予測するモデル
    Outputs:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: 最適化後の変数、bandgapの損失の履歴、Efの損失の履歴
    """
    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]
    atomic_distribution = nn.Parameter(
        mini_batch_inputs_dict["atomic_distribution"].clone().to("cuda"),
        requires_grad=True,
    )
    scaled_batch_abc = nn.Parameter(
        abc_scaler.scale(mini_batch_inputs_dict["batch_abc"].clone().to("cuda")),
        requires_grad=True,
    )
    scaled_batch_angle = nn.Parameter(
        angle_scaler.scale(mini_batch_inputs_dict["batch_angle"].clone().to("cuda")),
        requires_grad=True,
    )
    ox_mask_learnable_tensor_per_crystal = nn.Parameter(
        mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"].to("cuda"),
        requires_grad=True,
    )
    batch_dir_coords = nn.Parameter(
        mini_batch_inputs_dict["batch_dir_coords"].clone().to("cuda"),
        requires_grad=True,
    )
    radii_tensor = mini_batch_inputs_dict["radii_tensor"].to("cuda")

    # models
    bandgap_model_alignn = bandgap_model["ALIGNN"]
    e_form_model_alignn = e_form_model["ALIGNN"]
    bandgap_model_crystalformer = bandgap_model["Crystalformer"]
    e_form_model_crystalformer = e_form_model["Crystalformer"]

    # optimizer and scheduler
    (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    ) = define_optimizer_and_scheduler(
        lattice_lr=learning_rates[0],
        atom_lr=learning_rates[1],
        coords_lr=learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=atomic_distribution,
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        angle_optimization=angle_optimization,
        num_steps=num_steps,
        lattice_cycle=learning_rate_cycle[0],
        atom_cycle=learning_rate_cycle[1],
        coords_cycle=learning_rate_cycle[2],
    )

    # initialzation of history
    lattice_mat_history, coords_history, elememts_history = [], [], []
    abc_history, angle_history = [], []
    gap_history_alignn, e_form_history_alignn, t_loss_history_alignn = [], [], []
    (
        gap_history_crystalformer,
        e_form_history_crystalformer,
        t_loss_history_crystalformer,
    ) = (
        [],
        [],
        [],
    )
    VALID_ELEMENTS98 = create_element_list()

    for step_i in tqdm(range(num_steps), desc="Optimizing", leave=False):
        ### backpropのたびに計算する
        ### re-calculate atomic features, crystal structures, bondlength, bond angles
        ### ALIGNN
        optimization_targets = [
            scaled_batch_abc,
            scaled_batch_angle,
            batch_dir_coords,
            atomic_distribution,
            ox_mask_learnable_tensor_per_crystal,
        ]
        bandgap_pred_alignn, ef_pred_alignn, output_dict_alignn = (
            forward_propagation_ALIGNN(
                optimization_targets=optimization_targets,
                fixed_inputs=mini_batch_inputs_dict,
                scalers=scalers,
                temperature=dist_temp_sche[step_i],
                bandgap_model=bandgap_model_alignn,
                limit_coords_displacement=limit_coords_displacement,
                e_form_model=e_form_model_alignn,
                adding_noise_scale=adding_noise_scale,
            )
        )
        (
            total_loss_alignn,
            gap_loss_alignn,
            ef_loss_alignn,
            tolerance_loss_alignn,
            tolerance_alignn,
        ) = calculate_loss_from_output(
            bandgap_pred=bandgap_pred_alignn,
            ef_pred=ef_pred_alignn,
            sharpened_ox_mask=output_dict_alignn["sharpened_ox_mask"],
            normalized_dist=output_dict_alignn["normalized_dist"],
            site_ids=mini_batch_inputs_dict["site_ids"].to("cuda"),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[step_i],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

        ### Crystalformer
        optimization_targets = [
            scaled_batch_abc,
            scaled_batch_angle,
            batch_dir_coords,
            atomic_distribution,
            ox_mask_learnable_tensor_per_crystal,
        ]
        bandgap_pred_crystalformer, ef_pred_crystalformer, output_dict_crystalformer = (
            forward_propagation_Crystalformer(
                optimization_targets=optimization_targets,
                fixed_inputs=mini_batch_inputs_dict,
                scalers=scalers,
                temperature=dist_temp_sche[step_i],
                bandgap_model=bandgap_model_crystalformer,
                e_form_model=e_form_model_crystalformer,
                adding_noise_scale=adding_noise_scale,
                limit_coords_displacement=limit_coords_displacement,
            )
        )
        (
            total_loss_crystalformer,
            gap_loss_crystalformer,
            ef_loss_crystalformer,
            tolerance_loss_crystalformer,
            tolerance_crystalformer,
        ) = calculate_loss_from_output(
            bandgap_pred=bandgap_pred_crystalformer,
            ef_pred=ef_pred_crystalformer,
            sharpened_ox_mask=output_dict_crystalformer["sharpened_ox_mask"],
            normalized_dist=output_dict_crystalformer["normalized_dist"],
            site_ids=mini_batch_inputs_dict["site_ids"].to("cuda"),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[step_i],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

        ### total loss
        total_loss = total_loss_alignn + total_loss_crystalformer

        (
            optimizer_lattice.zero_grad(),
            optimizer_atom.zero_grad(),
            optimizer_coords.zero_grad(),
        )
        total_loss.backward()
        if step_i < step_of_x_opt_stop:
            optimizer_lattice.step(), optimizer_atom.step(), optimizer_coords.step()
        else:
            optimizer_lattice.step(), optimizer_coords.step()
        scheduler_lattice.step(), scheduler_atom.step(), scheduler_coords.step()

        # 損失の記録
        ### ALIGNN
        gap_history_alignn.append(bandgap_pred_alignn.detach().cpu().numpy())
        e_form_history_alignn.append(ef_pred_alignn.detach().cpu().numpy())
        t_loss_history_alignn.append(tolerance_alignn.detach().cpu().numpy())
        ### Crystalformer
        gap_history_crystalformer.append(
            bandgap_pred_crystalformer.detach().cpu().numpy()
        )
        e_form_history_crystalformer.append(
            ef_pred_crystalformer.detach().cpu().numpy()
        )
        t_loss_history_crystalformer.append(
            tolerance_crystalformer.detach().cpu().numpy()
        )

        if time_series_save:
            # 結晶の構成はALIGNNとCrystalformerで共通なので、ALIGNNの結果を使う
            lattice_mat_history.append(
                output_dict_alignn["lattice_vectors"].detach().cpu().numpy()
            )
            coords_history.append(
                output_dict_alignn["normed_batch_dir_coords"].detach().cpu().numpy()
            )
            abc_history.append(output_dict_alignn["batch_abc"].detach().cpu().numpy())
            angle_history.append(
                output_dict_alignn["batch_angle"].detach().cpu().numpy()
            )
            elements = [
                VALID_ELEMENTS98[atom_idx]
                for atom_idx in torch.argmax(
                    output_dict_alignn["normalized_dist"], dim=1
                )
            ]
            elememts_history.append(elements)

    # End of optimization
    print(
        f"ALIGNN he number of invalid crystral structures : {(torch.isnan(bandgap_pred_alignn)).detach().cpu().numpy().sum()}/{len(bandgap_pred_alignn)}"
    )
    print(
        f"Crystalformer he number of invalid crystral structures : {(torch.isnan(bandgap_pred_crystalformer)).detach().cpu().numpy().sum()}/{len(bandgap_pred_crystalformer)}"
    )
    optimized_mini_batch_inputs_dict = (
        mini_batch_inputs_dict | output_dict_alignn | output_dict_crystalformer
    )
    optimized_mini_batch_inputs_dict["atomic_distribution"] = atomic_distribution
    optimized_mini_batch_inputs_dict["scaled_batch_abc"] = scaled_batch_abc
    optimized_mini_batch_inputs_dict["scaled_batch_angle"] = scaled_batch_angle
    optimized_mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
        ox_mask_learnable_tensor_per_crystal
    )
    optimized_mini_batch_inputs_dict["batch_dir_coords"] = batch_dir_coords
    ### ALIGNN History
    optimized_mini_batch_inputs_dict["gap_history_alignn"] = np.transpose(
        np.array(gap_history_alignn).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["ef_history_alignn"] = np.transpose(
        np.array(e_form_history_alignn).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["t_history_alignn"] = np.transpose(
        np.array(t_loss_history_alignn).squeeze(), (1, 0)
    )
    ### Crystalformer History
    optimized_mini_batch_inputs_dict["gap_history_crystalformer"] = np.transpose(
        np.array(gap_history_crystalformer).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["ef_history_crystalformer"] = np.transpose(
        np.array(e_form_history_crystalformer).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["t_history_crystalformer"] = np.transpose(
        np.array(t_loss_history_crystalformer).squeeze(), (1, 0)
    )

    time_series = [
        lattice_mat_history,
        coords_history,
        abc_history,
        angle_history,
        elememts_history,
    ]

    return (
        optimized_mini_batch_inputs_dict,
        time_series,
    )


# In[ ]:


def evaluation_for_each_batch_Both(
    optimized_mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    crystal_system: Optional[str],
    limit_coords_displacement: Optional[float],
    e_form_min: Optional[float],
    Ef_coef_schedule: List[float],
    adding_noise_scale: Optional[float],
    onehot_temperature: float = 1e-8,
):
    """
    Evaluate the optimized solution for a batch of crystal structures using Crystalformer.
    """
    scaled_batch_abc = optimized_mini_batch_inputs_dict["scaled_batch_abc"]
    scaled_batch_angle = optimized_mini_batch_inputs_dict["scaled_batch_angle"]
    batch_dir_coords = optimized_mini_batch_inputs_dict["batch_dir_coords"]
    atomic_distribution = optimized_mini_batch_inputs_dict["atomic_distribution"]
    radii_tensor = optimized_mini_batch_inputs_dict["radii_tensor"].to("cuda")
    ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict[
        "ox_mask_learnable_tensor_per_crystal"
    ]
    size = optimized_mini_batch_inputs_dict["size"].to("cuda")
    batch_ori = optimized_mini_batch_inputs_dict["batch"].to("cuda")
    batch = batch_ori - batch_ori.min()  # batchを0から始まるように調整

    # models
    bandgap_model_alignn = bandgap_model["ALIGNN"]
    e_form_model_alignn = e_form_model["ALIGNN"]
    bandgap_model_crystalformer = bandgap_model["Crystalformer"]
    e_form_model_crystalformer = e_form_model["Crystalformer"]

    with torch.no_grad():
        ### ALIGNN
        # calculation with atomic distribution
        bandgap_dist_alignn, ef_dist_alignn, output_dict_alignn = (
            forward_propagation_ALIGNN(
                optimization_targets=[
                    scaled_batch_abc,
                    scaled_batch_angle,
                    batch_dir_coords,
                    atomic_distribution,
                    ox_mask_learnable_tensor_per_crystal,
                ],
                fixed_inputs=optimized_mini_batch_inputs_dict,
                scalers=scalers,
                temperature=onehot_temperature,
                bandgap_model=bandgap_model_alignn,
                limit_coords_displacement=limit_coords_displacement,
                e_form_model=e_form_model_alignn,
                adding_noise_scale=adding_noise_scale,
            )
        )
        _, loss_dist_alignn, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_dist_alignn,
            ef_pred=ef_dist_alignn,
            sharpened_ox_mask=output_dict_alignn["sharpened_ox_mask"],
            normalized_dist=output_dict_alignn["normalized_dist"],
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[-1],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )
        ### Crystalformer
        bandgap_dist_crystalformer, ef_dist_crystalformer, output_dict_crystalformer = (
            forward_propagation_Crystalformer(
                optimization_targets=[
                    scaled_batch_abc,
                    scaled_batch_angle,
                    batch_dir_coords,
                    atomic_distribution,
                    ox_mask_learnable_tensor_per_crystal,
                ],
                fixed_inputs=optimized_mini_batch_inputs_dict,
                scalers=scalers,
                temperature=onehot_temperature,
                bandgap_model=bandgap_model_crystalformer,
                e_form_model=e_form_model_crystalformer,
                adding_noise_scale=adding_noise_scale,
                limit_coords_displacement=limit_coords_displacement,
            )
        )
        _, loss_dist_crystalformer, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_dist_crystalformer,
            ef_pred=ef_dist_crystalformer,
            sharpened_ox_mask=optimized_mini_batch_inputs_dict["sharpened_ox_mask"],
            normalized_dist=optimized_mini_batch_inputs_dict["normalized_dist"],
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[-1],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )
        # combine the results
        # output_dict = output_dict_alignn | output_dict_crystalformer

        # calculation with one-hot atomic distribution
        max_val, onehot_x, onehot_atom_feat, onehot_ox_mask = calculate_onehot(
            output_dict_alignn,
            atom_feat_matrix=optimized_mini_batch_inputs_dict["atom_feat_matrix"].to(
                "cuda"
            ),
        )
        ### ALIGNN
        bandgap_onehot_alignn, ef_onehot_alignn = ALIGNN_prediction(
            bandgap_model=bandgap_model_alignn,
            e_form_model=e_form_model_alignn,
            g=optimized_mini_batch_inputs_dict["g"].to("cuda"),
            lg=optimized_mini_batch_inputs_dict["lg"].to("cuda"),
            atomic_features=onehot_atom_feat,  # one-hot atom features
            bondlength=output_dict_alignn["bondlength"],
            bond_angles=output_dict_alignn["bond_angles"],
        )
        _, loss_onehot_alignn, _, tolerance_loss_alignn, tolerance_alignn = (
            calculate_loss_from_output(
                bandgap_pred=bandgap_onehot_alignn,
                ef_pred=ef_onehot_alignn,
                sharpened_ox_mask=onehot_ox_mask,
                normalized_dist=onehot_x,
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                ef_coef=Ef_coef_schedule[-1],
                e_form_min=e_form_min,
                crystal_system=crystal_system,
                radii_tensor=radii_tensor,
            )
        )
        ### Crystalformer
        bandgap_onehot_crystalformer, ef_onehot_crystalformer = (
            Crystalformer_prediction(
                bandgap_model=bandgap_model_crystalformer,
                e_form_model=e_form_model_crystalformer,
                normalized_dist=onehot_x,
                pos=output_dict_crystalformer["pos"],
                batch=batch,
                trans_vec=output_dict_crystalformer["lattice_vectors"],
                size=size,
            )
        )
        (
            _,
            loss_onehot_crystalformer,
            _,
            tolerance_loss_crystalformer,
            tolerance_crystalformer,
        ) = calculate_loss_from_output(
            bandgap_pred=bandgap_onehot_crystalformer,
            ef_pred=ef_onehot_crystalformer,
            sharpened_ox_mask=onehot_ox_mask,
            normalized_dist=onehot_x,
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=Ef_coef_schedule[-1],
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

    # 結果の記録
    optimized_mini_batch_inputs_dict = (
        optimized_mini_batch_inputs_dict
        | output_dict_alignn
        | output_dict_crystalformer
    )
    ### ALIGNN, Crystalformer共通
    optimized_mini_batch_inputs_dict["max_val"] = max_val
    optimized_mini_batch_inputs_dict["onehot_x"] = onehot_x
    optimized_mini_batch_inputs_dict["onehot_ox_mask"] = onehot_ox_mask
    ### ALIGNN
    optimized_mini_batch_inputs_dict["bandgap_onehot_alignn"] = bandgap_onehot_alignn
    optimized_mini_batch_inputs_dict["bandgap_dist_alignn"] = bandgap_dist_alignn
    optimized_mini_batch_inputs_dict["ef_onehot_alignn"] = ef_onehot_alignn
    optimized_mini_batch_inputs_dict["ef_dist_alignn"] = ef_dist_alignn
    optimized_mini_batch_inputs_dict["loss_onehot_alignn"] = loss_onehot_alignn
    optimized_mini_batch_inputs_dict["loss_dist_alignn"] = loss_dist_alignn
    optimized_mini_batch_inputs_dict["tolerance_alignn"] = tolerance_alignn
    optimized_mini_batch_inputs_dict["tolerance_loss_alignn"] = tolerance_loss_alignn
    ### Crystalformer
    optimized_mini_batch_inputs_dict["bandgap_onehot_crystalformer"] = (
        bandgap_onehot_crystalformer
    )
    optimized_mini_batch_inputs_dict["bandgap_dist_crystalformer"] = (
        bandgap_dist_crystalformer
    )
    optimized_mini_batch_inputs_dict["ef_onehot_crystalformer"] = (
        ef_onehot_crystalformer
    )
    optimized_mini_batch_inputs_dict["ef_dist_crystalformer"] = ef_dist_crystalformer
    optimized_mini_batch_inputs_dict["loss_onehot_crystalformer"] = (
        loss_onehot_crystalformer
    )
    optimized_mini_batch_inputs_dict["loss_dist_crystalformer"] = (
        loss_dist_crystalformer
    )
    optimized_mini_batch_inputs_dict["tolerance_crystalformer"] = (
        tolerance_crystalformer
    )
    optimized_mini_batch_inputs_dict["tolerance_loss_crystalformer"] = (
        tolerance_loss_crystalformer
    )

    return optimized_mini_batch_inputs_dict


# In[ ]:


# ## 経時変化のプロットなど
# CrystalformerとALIGNN共通

# ### bandgapとformation energyの変化を可視化する

# In[ ]:


from jarvis.core.atoms import Atoms
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

# In[ ]:


def optimization_history_display_for_promising_candidates(
    gap_min: float,
    gap_max: float,
    npz_path: str,
    csv_path: str,
    model_name: str,
    num_display: int,
    history_img_path: str,
):
    d = np.load(npz_path, allow_pickle=True)
    # formation energyが低く、指定のバンドギャップを満たすものから順
    sorted_index = (
        pd.read_csv(csv_path)
        .sort_values(["loss_onehot", "ef_onehot"])["lattice_index"]
        .values
    )
    num_display = min(num_display, len(sorted_index))

    title_font_size, label_font_size, legend_font_size, tick_font_size = (
        16,
        14,
        14,
        14,
    )  # 目盛りのフォントサイズを追加
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列、図のサイズは12x5インチ

    colors_list = (
        list(mcolors.TABLEAU_COLORS.values())
        + list(mcolors.BASE_COLORS.values())
        + list(mcolors.CSS4_COLORS.values())
    )
    # Bandgap optimization historyのプロット（最初のサブプロット）
    if model_name != "Both":
        axs[0].plot(d["gap_history"][:, sorted_index[:num_display]])
        axs[0].hlines(
            gap_min,
            0,
            d["gap_history"].shape[0],
            linestyles="dashed",
            colors="black",
            label="target area",
        )
        axs[0].hlines(
            gap_max,
            0,
            d["gap_history"].shape[0],
            linestyles="dashed",
            colors="black",
        )
    else:
        for i in range(num_display):
            color = colors_list[i]
            axs[0].plot(
                d["gap_history_alignn"][:, sorted_index[i]], color=color, linestyle="-"
            )  # , label=f'alignn_{i}')
            axs[0].plot(
                d["gap_history_crystalformer"][:, sorted_index[i]],
                color=color,
                linestyle="-.",
            )
        axs[0].hlines(
            gap_min,
            0,
            d["gap_history_alignn"].shape[0],
            linestyles="dashed",
            colors="black",
            label="target area",
        )
        axs[0].hlines(
            gap_max,
            0,
            d["gap_history_alignn"].shape[0],
            linestyles="dashed",
            colors="black",
        )
        axs[0].plot([], color="gray", linestyle="-", label="alignn")
        axs[0].plot([], color="gray", linestyle="-.", label="crystalformer")

    axs[0].set_title(
        "Bandgap optimization history", fontsize=title_font_size
    )  # タイトルのフォントサイズ
    axs[0].set_xlabel("step", fontsize=label_font_size)  # x軸ラベルのフォントサイズ
    axs[0].set_ylabel(
        "bandgap (eV)", fontsize=label_font_size
    )  # y軸ラベルのフォントサイズ
    axs[0].tick_params(
        axis="both", labelsize=tick_font_size
    )  # x軸とy軸の目盛りのフォントサイズ

    axs[0].legend()  # 凡例の表示

    # Formation energy optimization historyのプロット（2番目のサブプロット）
    if model_name != "Both":
        axs[1].plot(d["ef_history"][:, sorted_index[:num_display]])
    else:
        for i in range(num_display):
            color = colors_list[i]
            axs[1].plot(
                d["ef_history_alignn"][:, sorted_index[i]], color=color, linestyle="-"
            )
            axs[1].plot(
                d["ef_history_crystalformer"][:, sorted_index[i]],
                color=color,
                linestyle="-.",
            )
    axs[1].set_title(
        "Formation energy optimization history", fontsize=title_font_size
    )  # タイトルのフォントサイズ
    axs[1].set_xlabel("step", fontsize=label_font_size)  # x軸ラベルのフォントサイズ
    axs[1].set_ylabel(
        "formation energy (eV/at.)", fontsize=label_font_size
    )  # y軸ラベルのフォントサイズ
    axs[1].tick_params(
        axis="both", labelsize=tick_font_size
    )  # x軸とy軸の目盛りのフォントサイズ

    plt.tight_layout()  # レイアウトの調整
    if os.path.dirname(history_img_path) != "":
        os.makedirs(os.path.dirname(history_img_path), exist_ok=True)
    plt.savefig(history_img_path)
    plt.close()
    figure = plt.figure(figsize=(10, 10))
    img = Image.open(history_img_path)
    plt.imshow(img)
    plt.show()


# ### 最適化された結晶構造をjarvis.core.atoms.Atomsオブジェクトとして保存するための辞書を構築する

# In[ ]:


def check_neurality_bondlength_and_save_structure(
    npz_path: str,
    csv_path: str,
    saved_dir: str,
    crystal_system: str,
    perovskite_size: str,
    acceptable_margin: float,
    neutral_check: bool = True,
) -> pd.DataFrame:
    """
    最適化された構造を保存する関数
    """

    # for large perovskite, neutral check is not necessary because of the large number of atoms
    if crystal_system == "perovskite" and perovskite_size in [
        "2x2x2",
        "3x3x3",
        "4x4x4",
    ]:
        print(
            f"<info> Perovskite size is large, so neutral check is not necessary. perovskite_size:{perovskite_size}"
        )
        neutral_check = False

    VALID_ELEMENTS98 = create_element_list()
    os.makedirs(f"{saved_dir}", exist_ok=True)

    d = np.load(npz_path, allow_pickle=True)

    # formation energyが低く、指定のバンドギャップを満たすものから順に並べる
    sorted_df = pd.read_csv(csv_path).sort_values(["loss_onehot", "ef_onehot"])

    sorted_index = sorted_df["lattice_index"].values
    opt_abc, opt_angle = compute_abc_angle(torch.tensor(d["lattice_vectors"]))
    coordinate_lattice_id = np.concatenate(
        [np.array([lattice_id] * num) for lattice_id, num in enumerate(d["num_atoms"])]
    )

    os.makedirs(f"{saved_dir}", exist_ok=True)

    neutralities = []
    for num_i, lattice_id in enumerate(
        tqdm(sorted_index, desc="save_structure", total=len(sorted_index))
    ):
        # exclude nan data
        if (
            np.isnan(d["dir_coords"][coordinate_lattice_id == lattice_id]).any()
            or np.isnan(d["lattice_vectors"][lattice_id]).any()
        ):
            continue

        # Neutral check
        elements_list = [
            VALID_ELEMENTS98[atom_idx]
            for atom_idx in np.argmax(
                d["onehot_x"][coordinate_lattice_id == lattice_id], axis=1
            )
        ]
        atoms, stoichs = [], []
        for _atom, _stoich in count_elements(elements_list).items():
            atoms.append(_atom)
            stoichs.append([_stoich])

        if neutral_check:
            is_neutral_smact, ox_states_smact = elec_neutral_check_SMACT(
                num_i, len(sorted_index), atoms, stoichs
            )
            is_neutral_cmn, all_elements_cmn, ox_states_cmn = (
                elec_neutral_check_SUPER_COMMON(
                    num_i, len(sorted_index), atoms, stoichs, True
                )
            )
            if crystal_system == "perovskite":
                sorted_ox_states = [tuple(sorted(tup)) for tup in ox_states_cmn]
                is_neutral_cmn = ((-1, -1, -1, 1, 2) in sorted_ox_states) or (
                    (-2, -2, -2, 2, 4) in sorted_ox_states
                )
            neutral_ok = is_neutral_cmn
        else:
            is_neutral_smact, ox_states_smact = np.nan, [np.nan]
            is_neutral_cmn, all_elements_cmn, ox_states_cmn = np.nan, [np.nan], [np.nan]
            neutral_ok = True

        # create Atoms object
        atom_dic = {
            "lattice_mat": [list(vec) for vec in d["lattice_vectors"][lattice_id]],
            "coords": [
                list(vec)
                for vec in d["dir_coords"][coordinate_lattice_id == lattice_id]
            ],
            "elements": elements_list,
            "abc": opt_abc[lattice_id].detach().cpu().numpy().tolist(),
            "angles": opt_angle[lattice_id].detach().cpu().numpy().tolist(),
            "cartesian": False,
            "props": [""] * len(d["dir_coords"][coordinate_lattice_id == lattice_id]),
        }
        assert len(atom_dic["elements"]) == len(atom_dic["coords"])
        assert len(atom_dic["elements"]) == len(atom_dic["props"])

        atoms_data = Atoms.from_dict(atom_dic)

        # bondlength check
        bondlength_check = check_nearest_neighbor(atoms_data)

        # add to data dictionary
        neutralities.append(
            {
                "original_fnames": d["original_fnames"][lattice_id],
                "is_neutral_smact": is_neutral_smact,
                "elements": atoms,
                "ox_states": ox_states_smact,
                "is_neutral_cmn": is_neutral_cmn,
                "elements_cmn": all_elements_cmn,
                "ox_states_cmn": ox_states_cmn,
                "minbond_less_than_0.5": bondlength_check,
                "one_atom": len(atoms) == 1,
            }
        )

        # structure saved
        fname = f"Optimized_based_on_{d['original_fnames'][lattice_id]}"
        atoms_data.write_poscar(os.path.join(saved_dir, f"{fname}"))
        loss_onehot = d["loss_onehot"][lattice_id]
        # print(f'loss_onehot: {loss_onehot}, is_neutral: {is_neutral_smact}, bondlength_check: {bondlength_check}')
        if loss_onehot < acceptable_margin and neutral_ok and bondlength_check:
            atoms_data.write_poscar(os.path.join(saved_dir, f"{fname}"))
            # Atoms.from_poscar(os.path.join(saved_dir, f'{fname}')).write_cif(f'./cif/{fname}.cif')

    return pd.DataFrame(neutralities)


# In[ ]:


# In[ ]:


# In[ ]:


import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.io import read
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


def draw_sphere(ax, center, radius, color):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=0.6)


def offset_text_position(center, radius, offset=1.2):
    # テキスト位置を球の半径よりも外側に配置
    return center + radius * np.array([offset, offset, offset])


def plot_crystals(poscar_dir: str, output_dir: str, text_size: int = 25):
    os.makedirs(output_dir, exist_ok=True)

    # 座標の最大値と最小値を初期化
    min_coords = np.array([np.inf, np.inf, np.inf])
    max_coords = np.array([-np.inf, -np.inf, -np.inf])

    # 全てのファイルを読み込んで座標の最大値と最小値を見つける
    for poscar_file in os.listdir(poscar_dir):
        if poscar_file.endswith(".vasp"):
            file_path = os.path.join(poscar_dir, poscar_file)
            atoms = read(file_path, format="vasp")
            for atom in atoms:
                min_coords = np.minimum(min_coords, atom.position)
                max_coords = np.maximum(max_coords, atom.position)

    # 軸の範囲を設定
    padding = 2.0  # 軸の余白
    x_limits = (np.min(min_coords) - padding, np.max(max_coords) + padding)
    y_limits = (np.min(min_coords) - padding, np.max(max_coords) + padding)
    z_limits = (np.min(min_coords) - padding, np.max(max_coords) + padding)

    # カラーマップを作成
    num_elements = 118  # 元素の種類の数
    cmap = plt.get_cmap("nipy_spectral")  # 色彩豊かなカラーマップを選択
    colors = cmap(np.linspace(0, 1, num_elements))

    file_list = os.listdir(poscar_dir)
    file_list.sort()
    # 指定されたディレクトリ内の全てのposcarファイルを処理
    for poscar_file in file_list:
        if poscar_file.endswith(".vasp"):
            file_path = os.path.join(poscar_dir, poscar_file)
            # POSCARファイルから構造を読み込み
            atoms = read(file_path, format="vasp")

            # サイズを調整し、レイアウトを変更
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(121, projection="3d")  # 3Dグラフ用の左側のサブプロット
            ax_text = fig.add_subplot(122)  # テキスト用の右側のサブプロット
            ax_text.axis("off")  # 軸を非表示にする

            # ユニットセルのエッジを描画
            cell = atoms.get_cell()
            # 基点
            origin = np.zeros(3)
            # 底面の頂点
            points = [origin, cell[0], cell[0] + cell[1], cell[1], origin]
            # 上面の頂点
            top_points = [point + cell[2] for point in points]

            # 底面と上面を描画
            for start, end in zip(points, points[1:]):
                ax.plot(*zip(start, end), linestyle="--", color="gray")
            for start, end in zip(top_points, top_points[1:]):
                ax.plot(*zip(start, end), linestyle="--", color="gray")

            # 垂直のエッジを描画
            for p1, p2 in zip(points, top_points):
                ax.plot(*zip(p1, p2), linestyle="--", color="gray")

            # ユニットセルの情報を取得
            atoms_dict = Atoms.from_poscar(file_path).to_dict()
            abc = atoms_dict["abc"]
            angles = atoms_dict["angles"]

            # 各原子の位置に原子をプロット
            for i, atom in enumerate(atoms):
                color = colors[
                    atom.number % num_elements
                ]  # 元素番号に基づいて色を割り当て
                # ax.scatter(atom.x, atom.y, atom.z, color=color, s=100)  # 原子のサイズと色を指定
                radius = covalent_radii[atom.number]  # 共有結合半径を使用
                draw_sphere(ax, atom.position, radius, color)
                text_pos = offset_text_position(atom.position, radius)
                # ax.text(text_pos[0], text_pos[1], text_pos[2], atom.symbol, fontsize=12, ha='center', color='white',
                #        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', edgecolor='none', alpha=0.5))
                # 座標を表示
                coord_text = f"No.{i + 1}"  # f"{atom.symbol}"
                ax.text(
                    atom.position[0],
                    atom.position[1],
                    atom.position[2] + radius * 1.2,
                    coord_text,
                    fontsize=12,
                    ha="center",
                    color="black",
                )

                # ユニットセルの情報をテキストとして追加
            unit_cell_info = (
                f"------ Crystal structure ------- \na={abc[0]:.2f}, b={abc[1]:.2f}, c={abc[2]:.2f}\n"
                f"α={angles[0]:.2f}°, β={angles[1]:.2f}°, γ={angles[2]:.2f}°\n"
                "\n"
            )

            # 原子情報を追加
            atom_info = "----- Atoms -----\n"
            for i, atom in enumerate(atoms):
                atom_info += f"No.{i + 1}: {atom.symbol} ({atom.position[0]:.2f}, {atom.position[1]:.2f}, {atom.position[2]:.2f})\n"

            # 全情報を表示
            full_text = unit_cell_info + atom_info
            ax_text.text(0.05, 0.95, full_text, va="top", ha="left", fontsize=text_size)

            # 軸の範囲を固定
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_zlim(z_limits)

            fig.tight_layout()  # For better spacing

            # 画像ファイルとして保存
            image_path = os.path.join(
                output_dir, poscar_file.replace(".vasp", "_structure.png")
            )
            plt.savefig(image_path)
            plt.close()
            # safe_display(Image.open(image_path))


def history_image(
    poscar_dir: str,
    output_dir: str,
    lattice_index: int,
    npz_path: str,
    gap_min: float,
    gap_max: float,
):
    title_font_size, label_font_size, legend_font_size, tick_font_size = (
        25,
        22,
        16,
        22,
    )  # 目盛りのフォントサイズを追加
    d = np.load(npz_path, allow_pickle=True)
    file_list = os.listdir(poscar_dir)
    file_list.sort()
    # 指定されたディレクトリ内の全てのposcarファイルを処理
    for poscar_file in file_list:
        if not poscar_file.endswith(".vasp"):
            continue
        match = re.search(r"\d+", poscar_file)
        time_step = int(match.group(0))

        # 履歴ファイルを呼び出し、現ステップ数以降にマスクをする
        gap_history = d["gap_history"][:, lattice_index]
        ef_history = d["ef_history"][:, lattice_index]
        gap_hist_max = np.max(gap_history) * 1.05 + 1.0
        ef_hist_min, ef_hist_max = (
            np.min(ef_history) * 1.05,
            np.max(ef_history) * 1.05 * 1.0,
        )

        # Setting NaN for the values after `time_step`
        gap_history[time_step:] = np.nan
        ef_history[time_step:] = np.nan

        fig, ax1 = plt.subplots()

        # Plotting gap_history on the primary y-axis
        color = "tab:blue"
        ax1.set_xlabel("Steps", fontsize=label_font_size)
        ax1.set_ylabel(r"$\mathrm{E_{bg}}$ (eV)", color=color, fontsize=label_font_size)
        ax1.plot(gap_history, label=r"$\mathrm{E_{bg}}$", color=color)
        ax1.tick_params(axis="x", labelsize=tick_font_size)
        ax1.tick_params(axis="y", labelcolor=color, labelsize=tick_font_size)
        ax1.set_ylim(0, gap_hist_max)  # Setting the y-axis limit for gap_history
        ax1.set_xlim(0, len(gap_history))  # Setting the x-axis limit
        ax1.hlines(
            gap_min,
            0,
            d["gap_history"].shape[0],
            linestyles="dashed",
            colors="black",
            label="band gap target area",
        )
        ax1.hlines(
            gap_max, 0, d["gap_history"].shape[0], linestyles="dashed", colors="black"
        )
        ax1.legend(loc="upper left", fontsize=legend_font_size)
        # Creating a second y-axis for ef_history
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel(
            r"$\mathrm{E_{form}}$ (eV/at.)", color=color, fontsize=label_font_size
        )
        ax2.plot(ef_history, label=r"$\mathrm{E_{form}}$ ", color=color)
        ax2.tick_params(axis="y", labelcolor=color, labelsize=tick_font_size)
        ax2.set_ylim(
            ef_hist_min, ef_hist_max
        )  # Setting the y-axis limit for ef_history
        plt.title("Optimization history", fontsize=title_font_size)
        ax2.legend(loc="upper right", fontsize=legend_font_size)

        fig.tight_layout()  # For better spacing

        image_path = os.path.join(
            output_dir, poscar_file.replace(".vasp", "_history.png")
        )
        plt.savefig(image_path)
        plt.close()


# In[ ]:


# In[ ]:


import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_gif_files(image_dir, gif_output_dir, gif_name):
    """
    結晶構造の画像と最適化履歴の画像を組み合わせてGIFを作成する関数
    """
    os.makedirs(gif_output_dir, exist_ok=True)

    history_img_paths = glob.glob(os.path.join(image_dir, "*_history.png"))
    for history_path in history_img_paths:
        structure_path = history_path.replace("_history.png", "_structure.png")
        img1 = mpimg.imread(history_path)
        img2 = mpimg.imread(structure_path)

        # GridSpecを使用して画像の表示サイズを調整
        fig = plt.figure(figsize=(12, 4))
        gs = GridSpec(1, 5, figure=fig)  # 3カラムのグリッドを定義

        # 最初の画像を表示
        ax1 = fig.add_subplot(gs[0, 0:2])  # 1つ目のカラムを使用
        ax1.imshow(img1, aspect="equal")
        ax1.axis("off")

        # 二つ目の画像を表示（より広い範囲に表示）
        ax2 = fig.add_subplot(gs[0, 2:])  # 2つ目と3つ目のカラムを結合して使用
        ax2.imshow(img2, aspect="equal")
        ax2.axis("off")
        fig.tight_layout()  # For better spacing

        image_path = os.path.join(
            gif_output_dir, history_path.split("/")[-1].replace("_history.png", ".png")
        )
        plt.savefig(image_path)
        plt.close()
        # display(Image.open(image_path))

    ### GIFを作る
    # 画像ファイルのパスをリストとして取得
    image_files = sorted(glob.glob(f"{gif_output_dir}/*.png"))

    # 画像ファイルを開いてリストに追加
    images = [Image.open(image) for image in image_files]

    # GIFを保存
    output_path = os.path.join(gif_output_dir, f"{gif_name}.gif")
    images[0].save(
        output_path, save_all=True, append_images=images[1:], duration=300, loop=0
    )

    # 不要なファイルを削除
    history_img_paths = glob.glob(os.path.join(image_dir, "*_history.png"))
    for history_path in history_img_paths:
        structure_path = history_path.replace("_history.png", "_structure.png")
        os.remove(history_path)
        os.remove(structure_path)
    image_files = sorted(glob.glob(f"{gif_output_dir}/*.png"))
    for image_file in image_files:
        os.remove(image_file)


# In[ ]:


import re
from typing import List


def create_optimization_GIF_file(
    time_series: List[np.ndarray],
    gap_min: float,
    gap_max: float,
    npz_path: str,
    csv_path: str,
    num_steps: int,
    saved_gif_dir: str,
    tmp_output_dir="tmp_output_dir",
    num_saves: int = 5,
    save_time_step: int = 5,
):
    """
    最適化されていく構造のGIFを作成する関数
    """
    (
        lattice_mat_history,
        coords_history,
        abc_history,
        angle_history,
        elememts_history,
    ) = time_series
    d = np.load(npz_path, allow_pickle=True)

    coordinate_lattice_id = np.concatenate(
        [np.array([lattice_id] * num) for lattice_id, num in enumerate(d["num_atoms"])]
    )
    sorted_index = (
        pd.read_csv(csv_path)
        .sort_values(["loss_onehot", "ef_onehot"])["lattice_index"]
        .values
    )

    for lattice_id in tqdm(sorted_index[:num_saves]):
        time_poscar_dir = f"./crystal{lattice_id}_time_series"
        os.makedirs(time_poscar_dir, exist_ok=True)

        # 各時刻の最適化途中の結晶構造を保存する
        for time_step in range(num_steps):
            if (not time_step % save_time_step == 0) and (
                not time_step == num_steps - 1
            ):
                continue

            lattice_mat = lattice_mat_history[time_step]
            coords = coords_history[time_step]
            opt_abc = abc_history[time_step]
            opt_angle = angle_history[time_step]
            elements = elememts_history[time_step]
            atom_dic = {
                "lattice_mat": [list(vec) for vec in lattice_mat[lattice_id]],
                "coords": [
                    list(vec) for vec in coords[coordinate_lattice_id == lattice_id]
                ],
                "elements": np.array(elements)[
                    coordinate_lattice_id == lattice_id
                ].tolist(),
                "abc": opt_abc[lattice_id].tolist(),
                "angles": opt_angle[lattice_id].tolist(),
                "cartesian": False,
                "props": [""] * len(coords[coordinate_lattice_id == lattice_id]),
            }
            assert len(atom_dic["elements"]) == len(atom_dic["coords"])
            assert len(atom_dic["elements"]) == len(atom_dic["props"])
            formatted_time_step = "{:04}".format(time_step)
            Atoms.from_dict(atom_dic).write_poscar(
                os.path.join(time_poscar_dir, f"step_{formatted_time_step}.vasp")
            )

        # GIFを作成
        time_poscar_dir = f"./crystal{lattice_id}_time_series"
        gif_name = f"crystal{lattice_id}_time_series"
        os.makedirs(tmp_output_dir, exist_ok=True)
        os.makedirs(saved_gif_dir, exist_ok=True)
        plot_crystals(time_poscar_dir, tmp_output_dir)
        history_image(
            time_poscar_dir, tmp_output_dir, lattice_id, npz_path, gap_min, gap_max
        )
        create_gif_files(tmp_output_dir, saved_gif_dir, gif_name)
        shutil.rmtree(tmp_output_dir)
        shutil.rmtree(time_poscar_dir)


# ## 最近傍原子間距離の計算

# In[ ]:


def check_nearest_neighbor(atoms_data: Atoms, r: float = 0.5, verbose: int = 0) -> bool:
    """
    Checks for any atom pairs within a specified distance in a supercell.

    Args:
        atoms_data (Atoms): An instance of Atoms containing atomic data.
        r (float): Radius to check for neighbor proximity. Default is 0.5.
        verbose (int): Verbosity level. A non-zero value prints detailed bond information.

    Returns:
        bool: True if no atom pairs are found within the specified distance, False otherwise.
    """
    try:
        # Generate supercell
        sc_atoms_data = atoms_data.make_supercell_matrix([2, 2, 2])
        coords = (
            sc_atoms_data.coords
        )  # Assuming this now correctly handles supercell coordinates

        # Compute distance matrix efficiently
        from scipy.spatial import distance_matrix

        dist_matrix = distance_matrix(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)

        # get original atom ids
        atom_ids = np.array(
            [[atoms_data.elements.index(e)] * (2 * 2 * 2) for e in atoms_data.elements]
        ).reshape(-1)

        # Identify pairs with distance less than r
        close_pairs = np.where(dist_matrix < r)

        if verbose > 0:
            for _row, _col in zip(*close_pairs):
                coord0 = coords[_row]
                coord1 = coords[_col]
                print(
                    f"bond between: Atom-{atom_ids[_row]} {sc_atoms_data.elements[_row]}({coord0[0]:.3f},{coord0[1]:.3f},{coord0[2]:.3f}) ----- Atom-{atom_ids[_col]} {sc_atoms_data.elements[_col]}({coord1[0]:.3f},{coord1[1]:.3f},{coord1[2]:.3f}), distance:{dist_matrix[_row, _col]:.3f}"
                )
                original_coord0 = atoms_data.coords[atom_ids[_row]]
                original_coord1 = atoms_data.coords[atom_ids[_col]]
                print(
                    f"original coord:  Atom-{atom_ids[_row]} {atoms_data.elements[atom_ids[_row]]}({original_coord0[0]:.3f},{original_coord0[1]:.3f},{original_coord0[2]:.3f}) ----- Atom-{atom_ids[_col]} {atoms_data.elements[atom_ids[_col]]}({original_coord1[0]:.3f},{original_coord1[1]:.3f},{original_coord1[2]:.3f})"
                )
                print("")

        return not any(dist_matrix.flatten() < r)

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


# ## SMACTを用いた電気的中性かの確認

# In[ ]:


import itertools
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jarvis_data
from smact import Element, element_dictionary, neutral_ratios


def count_elements(elements_list):
    elements = {}
    for element in elements_list:
        if element not in elements:
            elements[element] = 0
        elements[element] += 1
    return elements


def elec_neutral_check_SMACT(
    num_i: int, total: int, elements: List[str], stoichs: List[List[int]]
) -> Tuple[bool, Optional[Tuple[int]]]:
    """
    Check for electrical neutrality using SMACT by evaluating possible oxidation states combinations.

    Args:
        num_i (int): Index of the structure (for tqdm display)
        total (int): Total number of structures. (for tqdm display)
        elements (List[str]): List of element symbols.
        stoichs (List[List[int]]): List of lists containing stoichiometries.

    Returns:
        Tuple[bool, Optional[Tuple[int]]]: A tuple where the first element is a boolean indicating
                                           whether the input is electrically neutral, and the second
                                           element is a tuple of oxidation states that make it neutral
                                           (or None if no neutral combination is found).

    Examples:
        >>> elec_neutral_check_SMACT(5, 10, elements=['Ti', 'O'], stoichs=[[1], [2]])
        (True, (4, -2))
        >>> elec_neutral_check_SMACT(5, 10, elements = ['Ti', 'Al', 'O'], stoichs = [[1],[1],[1]])
        (True, (4, 3, -6))
        >>> elec_neutral_check_SMACT(5, 10, elements=['H', 'O'], stoichs=[[1], [2]])
        (False, None)
    """
    space = element_dictionary(elements)
    els = [space[element] for element in elements]
    ox_combos = [el.oxidation_states for el in els]
    lengths = np.array([len(sublist) for sublist in ox_combos])
    product_of_lengths = np.prod(lengths)
    for ox_states in tqdm(
        itertools.product(*ox_combos),
        total=product_of_lengths,
        leave=False,
        desc=f"neutral check ({num_i + 1}/{total}) by SMACT",
    ):
        cn_e, cn_r = neutral_ratios(ox_states, stoichs=stoichs, threshold=None)
        if cn_e:
            return True, ox_states
    return False, None


# In[ ]:


def elec_neutral_check_PMG_ICSD(
    num_i: int, total: int, elements: List[str], stoichs: List[List[int]]
) -> Tuple[bool, Optional[Tuple[int]]]:
    """
    Check for electrical neutrality using PyMatGen icsd_oxidation_states method by evaluating possible oxidation states combinations.

    Args:
        num_i (int): Index of the structure (for tqdm display)
        total (int): Total number of structures. (for tqdm display)
        elements (List[str]): List of element symbols.
        stoichs (List[List[int]]): List of lists containing stoichiometries.

    Returns:
        Tuple[bool, Optional[Tuple[int]]]: A tuple where the first element is a boolean indicating
                                           whether the input is electrically neutral, and the second
                                           element is a tuple of oxidation states that make it neutral
                                           (or None if no neutral combination is found).

    Examples:
        >>> elec_neutral_check_PMG_ICSD(5, 10, elements=['Ti', 'O'], stoichs=[[1], [2]])
        (True, , ['Ti', 'O', 'O'], (4, -2, -2)))
        >>> elec_neutral_check_PMG_ICSD(5, 10, elements = ['Ti', 'Al', 'O'], stoichs = [[1],[1],[1]])
        (False, ['Ti', 'Al', 'O'], None)
        >>> elec_neutral_check_PMG_ICSD(5, 10, elements=['H', 'O'], stoichs=[[1], [2]])
        (False, ['H', 'O', 'O'], None)
    """
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem] * stoi[0])
    ox_combos = [Element_pmg(elem).icsd_oxidation_states for elem in all_elements]
    lengths = np.array([len(sublist) for sublist in ox_combos])
    product_of_lengths = np.prod(lengths)
    for ox_states in tqdm(
        itertools.product(*ox_combos),
        total=product_of_lengths,
        leave=False,
        desc=f"neutral check ({num_i + 1}/{total}) by PMG",
    ):
        if sum(ox_states) == 0:
            return True, all_elements, ox_states
    return False, all_elements, None


# In[ ]:


def elec_neutral_check_SUPER_COMMON(
    num_i: int,
    total: int,
    elements: List[str],
    stoichs: List[List[int]],
    return_all_ox_states: bool = False,
) -> Tuple[bool, Optional[Tuple[int]]]:
    """
    Check for electrical neutrality using PyMatGen icsd_oxidation_states method by evaluating possible oxidation states combinations.

    Args:
        num_i (int): Index of the structure (for tqdm display)
        total (int): Total number of structures. (for tqdm display)
        elements (List[str]): List of element symbols.
        stoichs (List[List[int]]): List of lists containing stoichiometries.
        return_all_ox_states (bool): Whether to return all possible oxidation states combinations.

    Returns:
        Tuple[bool, Optional[Tuple[int]]]: A tuple where the first element is a boolean indicating
                                           whether the input is electrically neutral, and the second
                                           element is a tuple of oxidation states that make it neutral
                                           (or None if no neutral combination is found).

    Examples:
        >>> elec_neutral_check_SUPER_COMMON(5, 10, elements=['Ti', 'O'], stoichs=[[1], [2]])
        (True, , ['Ti', 'O', 'O'], (4, -2, -2)))
        >>> elec_neutral_check_SUPER_COMMON(5, 10, elements = ['Ti', 'Al', 'O'], stoichs = [[1],[1],[1]])
        (False, ['Ti', 'Al', 'O'], None)
        >>> elec_neutral_check_SUPER_COMMON(5, 10, elements=['He', 'O'], stoichs=[[1], [2]])
        (False, ['H', 'O', 'O'], None)
    """
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem] * stoi[0])
    ox_combos = [
        list(
            set(Element_pmg(elem).icsd_oxidation_states)
            & set(Element_pmg(elem).oxidation_states)
            & set(Element(elem).oxidation_states)
            & set(Element_pmg(elem).common_oxidation_states)
        )
        for elem in all_elements
    ]

    # check excluding non-oxidation state elements
    if any([len(ox) == 0 for ox in ox_combos]):
        return False, all_elements, None

    lengths = np.array([len(sublist) for sublist in ox_combos])
    product_of_lengths = np.prod(lengths)

    if return_all_ox_states:
        all_neutral_ox_states = []
        for ox_states in tqdm(
            itertools.product(*ox_combos),
            total=product_of_lengths,
            leave=False,
            desc=f"neutral check ({num_i + 1}/{total}) by PMG",
        ):
            if sum(ox_states) == 0:
                all_neutral_ox_states.append(ox_states)
        return len(all_neutral_ox_states) > 0, all_elements, all_neutral_ox_states

    else:
        for ox_states in tqdm(
            itertools.product(*ox_combos),
            total=product_of_lengths,
            leave=False,
            desc=f"neutral check ({num_i + 1}/{total}) by PMG",
        ):
            if sum(ox_states) == 0:
                return True, all_elements, ox_states

        return False, all_elements, None


# ### ペロブスカイトの構造になっているかを確認する関数

# In[ ]:


def perovskite_coordinate_check(
    init_coords: np.ndarray,
    dir_coords: np.ndarray,
    size: np.ndarray,
    limit_of_displacement: float = 0.15,
) -> np.ndarray:
    """
    ペロブスカイト構造の最適化において、原子の座標が一定の範囲内に収まっているかを確認する関数。
    """
    displacement = np.minimum(
        np.abs(init_coords - dir_coords),
        np.abs(init_coords - dir_coords + 1),
        np.abs(init_coords - dir_coords - 1),
    )
    assert (size[0] == size).all()
    return (
        (displacement <= limit_of_displacement)
        .all(axis=1)
        .reshape(len(size), size[0])
        .all(axis=1)
    )


# In[ ]:


# In[ ]:


# In[ ]:


# ## 実験の補助関数

# In[ ]:


import os
import shutil
from typing import NoReturn


def remove_files_in_directory(directory: str) -> NoReturn:
    """Deletes all files and subdirectories within the specified directory.

    This function recursively deletes all files, symbolic links, and subdirectories within the specified directory.
    It outputs an error message if it fails to delete a file, symbolic link, or directory.

    Args:
        directory (str): The path to the directory from which files and directories are to be deleted.

    Raises:
        Exception: Outputs an error message if there is an issue during file deletion.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively remove directory
        except Exception as e:
            print(f"Failed to delete. File: {file_path}. Error: {e}")


# In[ ]:


def define_initial_dataset_dir_path(
    crystal_system: Optional[str],
    initial_dataset,
    neutral: bool,
    max_atoms: Optional[int] = 10,
    perovskite_mode: Optional[str] = None,
    perovskite_size: Optional[str] = None,
):
    raw_data_dir = os.path.join(get_master_dir(), "data/raw_data")

    if max_atoms is None:
        max_atoms = 100000000
        max_strings = ""
    else:
        max_strings = f"_max_{max_atoms}atoms"

    if neutral == "neutral":
        suffix = "_neutral"
    elif neutral == "common":
        suffix = "_neutral_common"
    elif neutral == "super_common":
        suffix = "_neutral_super_common"
    elif not neutral:
        suffix = ""
    else:
        raise ValueError(f"Invalid value for neutral: {neutral}")

    if crystal_system is None:
        dir_path = os.path.join(
            raw_data_dir,
            f"initial_candidates_from_{initial_dataset}{max_strings}{suffix}",
        )
        dataset_name = f"{initial_dataset}{max_strings}{suffix}"
    elif crystal_system == "perovskite":
        if perovskite_size is None:
            size_str = ""
        else:
            perovskite_size in ["2x2x2", "3x3x3", "4x4x4"]
            size_str = f"_{perovskite_size}"
        dataset_name = f"initial_{perovskite_mode}_perovskite{size_str}"
        dir_path = os.path.join(raw_data_dir, dataset_name)

    else:
        raise NotImplementedError
    assert os.path.exists(dir_path), f"{dir_path} does not exist."

    return dir_path, dataset_name


# In[ ]:


def define_experiment_name(
    model_name: str,
    dataset_name: str,
    optimize_mode: str,
    target_bandgap: float,
    bandgap_margin: float,
    use_formation_energy: bool,
    e_form_min: Optional[float],
    learning_rates: List[float],
    learning_cycles: List[float],
    adding_noise_scale: Optional[float],
    num_steps: int,
    num_graph_update: Optional[float],
):
    lattice_lr, atom_lr, coords_lr = learning_rates
    lattice_cycle, atom_cycle, coords_cycle = learning_cycles
    if lattice_cycle < 1:
        lattice_cycle = 0
    if atom_cycle < 1:
        atom_cycle = 0
    if coords_cycle < 1:
        coords_cycle = 0

    if e_form_min is None:
        e_form_suffix = ""
    else:
        e_form_suffix = f"-min{e_form_min:.4f}"

    if adding_noise_scale is not None:
        noise_suffix = f"__noise{adding_noise_scale:.4f}"
    else:
        noise_suffix = ""

    if num_graph_update is not None and model_name == "ALIGNN":
        model_name = f"{model_name}up{num_graph_update}"

    exp_name = f"{model_name}__{dataset_name}__bg{target_bandgap:.2f}pm{bandgap_margin:.2f}__EfCoef{float(use_formation_energy)}{e_form_suffix}__Atomlr{atom_lr}_c{atom_cycle}__Latticelr{lattice_lr}_c{lattice_cycle}__Coordslr{coords_lr}_c{coords_cycle}__ns{num_steps}{noise_suffix}/{optimize_mode}"
    return exp_name


# In[ ]:


# In[ ]:


def save_optimization_results(
    npz_path: str,
    optimized_dict_list: List[Dict[str, torch.Tensor]],
    model_name: str,
    crystal_system: Optional[str],
    test_noise_scales: List[float],
):
    common_data_dict = {
        "lattice_vectors": torch.cat(
            [opt_dict["lattice_vectors"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "dir_coords": torch.cat(
            [
                opt_dict["normed_batch_dir_coords"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_coords": torch.cat(
            [opt_dict["init_coords"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "scaled_batch_abc": torch.cat(
            [opt_dict["scaled_batch_abc"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "scaled_batch_angle": torch.cat(
            [
                opt_dict["scaled_batch_angle"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "ox_states_used_mask": torch.cat(
            [
                opt_dict["ox_states_used_mask"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "atomic_mask": torch.cat(
            [opt_dict["atomic_mask"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "sharpened_ox_mask": torch.cat(
            [
                opt_dict["sharpened_ox_mask"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "atomic_distribution": torch.cat(
            [
                opt_dict["atomic_distribution"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "ox_mask_learnable_tensor_per_crystal": torch.cat(
            [
                opt_dict["ox_mask_learnable_tensor_per_crystal"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "normalized_dist": torch.cat(
            [opt_dict["normalized_dist"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "onehot_x": torch.cat(
            [opt_dict["onehot_x"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "batch_abc": torch.cat(
            [opt_dict["batch_abc"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "batch_angle": torch.cat(
            [opt_dict["batch_angle"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_abc": torch.cat(
            [opt_dict["init_abc"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_angles": torch.cat(
            [opt_dict["init_angles"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "size": torch.cat(
            [opt_dict["size"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "site_ids": torch.cat(
            [opt_dict["site_ids"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "num_atoms": torch.cat(
            [opt_dict["size"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "original_fnames": np.concatenate(
            [opt_dict["fnames"] for opt_dict in optimized_dict_list]
        ),
    }
    if model_name == "Both":
        model_suffixes = ["_crystalformer", "_alignn"]
    else:
        model_suffixes = [""]

    for model_sfx in model_suffixes:
        # loss etc.
        for key in [
            "bandgap_onehot",
            "bandgap_dist",
            "ef_onehot",
            "ef_dist",
            "loss_onehot",
            "tolerance",
            "tolerance_loss",
        ]:
            common_data_dict[key + model_sfx] = (
                torch.cat(
                    [
                        opt_dict[key + model_sfx].squeeze()
                        for opt_dict in optimized_dict_list
                    ]
                )
                .detach()
                .cpu()
                .numpy()
            )

        # History
        for key in ["gap_history", "ef_history", "t_history"]:
            common_data_dict[key + model_sfx] = np.concatenate(
                [
                    opt_dict[key + model_sfx].squeeze()
                    for opt_dict in optimized_dict_list
                ]
            ).T

        # noise scale test
        for noise_scale in test_noise_scales:
            for key in ["band_mean", "band_std", "e_form_mean", "e_form_std"]:
                common_data_dict[f"{key}{model_sfx}_noise{noise_scale}"] = (
                    torch.cat(
                        [
                            opt_dict[f"{key}{model_sfx}_noise{noise_scale}"].squeeze()
                            for opt_dict in optimized_dict_list
                        ]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

    # important results
    if model_name == "Both":
        common_data_dict["loss_onehot"] = (
            common_data_dict["loss_onehot_alignn"]
            + common_data_dict["loss_onehot_crystalformer"]
        )
        common_data_dict["bandgap_onehot"] = (
            common_data_dict["bandgap_onehot_alignn"]
            + common_data_dict["bandgap_onehot_crystalformer"]
        ) / 2
        common_data_dict["ef_onehot"] = (
            common_data_dict["ef_onehot_alignn"]
            + common_data_dict["ef_onehot_crystalformer"]
        ) / 2
        if crystal_system == "perovskite":
            torch.testing.assert_close(
                common_data_dict["tolerance_alignn"],
                common_data_dict["tolerance_crystalformer"],
            )
        common_data_dict["tolerance"] = common_data_dict["tolerance_alignn"]

    np.savez(npz_path, **common_data_dict)


# In[ ]:


def extract_conditions(experiment_name):
    """
    実験名の文字列から各条件を抽出し、辞書形式で返す関数。

    Args:
    experiment_name (str): 実験名の文字列

    Returns:
    dict: 条件が格納された辞書。マッチしない場合は空の辞書を返す。
    """
    # 条件を抽出するための正規表現パターン
    # Splitting the name into parts based on '__' and other identifiable markers
    experiment_name, optimize_settings = experiment_name.split("/")
    parts = experiment_name.split("__")
    print(parts)
    model_name = parts[0]
    dataset_name = parts[1]
    bg_info = parts[2].split("bg")[1].split("pm")
    target_bandgap = float(bg_info[0])
    bandgap_margin = float(bg_info[1])
    use_formation_energy = float(parts[3].split("EfCoef")[1])
    atom_info = parts[4].split("Atomlr")[1].split("_c")
    atom_lr = float(atom_info[0])
    atom_cycle = float(atom_info[1])
    lattice_info = parts[5].split("Latticelr")[1].split("_c")
    lattice_lr = float(lattice_info[0])
    lattice_cycle = float(lattice_info[1])
    coords_info = parts[6].split("Coordslr")[1].split("_c")
    coords_lr = float(coords_info[0])
    coords_cycle = float(coords_info[1])
    num_steps = int(parts[7].split("ns")[1])

    # Constructing the dictionary
    result = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "target_bandgap": target_bandgap,
        "bandgap_margin": bandgap_margin,
        "use_formation_energy": use_formation_energy,
        "atom_lr": atom_lr,
        "atom_cycle": atom_cycle,
        "lattice_lr": lattice_lr,
        "lattice_cycle": lattice_cycle,
        "coords_lr": coords_lr,
        "coords_cycle": coords_cycle,
        "num_steps": num_steps,
        "optimize_settings": optimize_settings,
    }

    # Handling optional noise component if exists
    if len(parts) > 8:
        noise_scale = float(parts[8].split("noise")[1])
        result["adding_noise_scale"] = noise_scale

    return result


# In[ ]:


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


# ### TPE optimization
#
# #### 良い解を使う設定の最適化

# In[ ]:


import optuna
from optuna.samplers import TPESampler


class TPE_optimization:
    def __init__(
        self,
        model_name: str,
        num_steps: int,
        bandgap_margin: float,
        target_bandgap: float,
        exp_id: int,
        crystal_system: str,
        perovskite_size: Optional[str],
    ):
        self.model_name = model_name
        self.num_steps = num_steps
        self.bandgap_margin = bandgap_margin
        self.target_bandgap = target_bandgap
        self.exp_id = exp_id
        self.crystal_system = crystal_system
        self.perovskite_size = perovskite_size
        if self.crystal_system == None:
            assert self.perovskite_size is None
            self.limit_coords_displacement = None
            self.num_candidates = self.num_batch_crystal = 256
        elif self.crystal_system == "perovskite":
            self.limit_coords_displacement = 0.15
            if self.perovskite_size in ["2x2x2", "3x3x3", "4x4x4"]:
                if model_name == "ALIGNN":
                    self.num_candidates = self.num_batch_crystal = 128
                elif model_name == "Crystalformer":
                    self.num_candidates = self.num_batch_crystal = 32
                else:
                    raise NotImplementedError
            else:
                self.num_candidates = self.num_batch_crystal = 256

        self.all_reusult_dict = []

    def objective_function(self, trial):
        return main_experiment(
            trial,
            all_reusult_dict=self.all_reusult_dict,
            model_name=self.model_name,
            num_steps=self.num_steps,
            bandgap_margin=self.bandgap_margin,
            target_bandgap=self.target_bandgap,
            test_noise_scales=[],
            crystal_system=self.crystal_system,
            perovskite_size=self.perovskite_size,
            limit_coords_displacement=self.limit_coords_displacement,
            exp_id=self.exp_id,
            num_candidate=self.num_candidates,
            num_batch_crystal=self.num_batch_crystal,
        )


# In[ ]:


def Hyperparameter_TPE_optimization(
    args: argparse.Namespace, n_trials: int = 200, work_dir: str = "./"
):
    target_bandgap = args.target_bandgap
    bandgap_margin = args.bandgap_margin
    num_steps = args.num_steps
    exp_id = args.experiment_id
    model_name = args.model_name
    crystal_system = args.crystal_system
    perovskite_size = args.perovskite_size

    workspace_dir = os.path.join(work_dir, f"workspace{exp_id}")
    os.makedirs(workspace_dir, exist_ok=True)
    os.chdir(workspace_dir)

    sampler = optuna.samplers.TPESampler()
    tpe_opt = TPE_optimization(
        model_name=model_name,
        crystal_system=crystal_system,
        perovskite_size=perovskite_size,
        num_steps=num_steps,
        bandgap_margin=bandgap_margin,
        target_bandgap=target_bandgap,
        exp_id=exp_id,
    )

    study = optuna.create_study(direction="maximize", sampler=sampler)
    # 初期値を指定
    study.enqueue_trial(
        {
            "atom_lr_pow": 0.75,
            "lattice_lr_pow": -2.8,
            "coords_lr_pow": -1.8,
            "cm_str": "None",
            "mutation_noise": 0.0,
            "top_ratio": 0.0,
            "group_C_use_rate": 1.0,
            "atom_dist_init": False,
        }
    )

    study.optimize(tpe_opt.objective_function, n_trials=n_trials)


# In[ ]:


# In[ ]:


# # main.py

# In[ ]:


# if __name__ == '__main__':
def main_experiment(
    trial=None,
    all_reusult_dict=Optional[List[Dict[str, Any]]],
    num_steps: int = 200,
    model_name="Crystalformer",  #'Crystalformer' or 'ALIGNN' or 'Both'
    optimize_mode="pytorch_optimizer",
    bandgap_margin: float = 0.01,
    num_candidate: int = 128,
    num_batch_crystal: int = 128,
    target_bandgap: Optional[float] = None,
    perovskite_size: Optional[str] = None,
    test_noise_scales=[0.01, 0.005, 0.001],
    crystal_system: Optional[str] = None,
    exp_id: int = 0,
    atom_lr: float = 2.0,
    lattice_lr: float = 0.003,
    coords_lr: float = 0.005,
    atom_cycle: int = 1,
    lattice_cycle: int = 1,
    coords_cycle: int = 1,
    E_form_criteria: float = -0.5,
    limit_coords_displacement=0.15,
    angle_range=(60, 120),  # 30, 150
    abc_range=(2.0, 10.0),  # 2.0, 10.0
    max_ox_states: int = 10,
    test_mode: bool = False,
    ALIGNN_update_step: Optional[List[int]] = None,
    initialization: str = "Struct-AtomicMask",
    method: str = "grad",
    seed: int = 42,
    exp_name: Optional[str] = None,
    output_root: Optional[str] = None,
):
    """
    #デバッグ用の引数設定
    trial = None
    all_reusult_dict = None
    num_steps = 200
    bandgap_margin = 0.04
    num_candidate = 256
    num_batch_crystal = 256
    crystal_system = None
    test_noise_scales = []
    model_name = 'Crystalformer' #'Crystalformer' or 'ALIGNN' or 'Both'
    E_form_criteria = -0.5

    target_bandgap = 2.0
    perovskite_size = None
    angle_range = (30, 150)
    abc_range = (2.0, 10.0)
    exp_id = 0
    atom_lr = 6.0
    lattice_lr = 0.02
    coords_lr = 0.03
    atom_cycle = 1
    lattice_cycle = 1
    coords_cycle = 1
    limit_coords_displacement = None
    ALIGNN_update_step = calculate_update_steps(num_steps=200, num_update=4)
    test_mode = False
    """

    if not exp_name or Path(exp_name).name != exp_name or exp_name in {".", ".."}:
        raise ValueError("exp_name must be a single directory name")
    if not 2 <= num_candidate <= 256 or not 2 <= num_batch_crystal <= num_candidate:
        raise ValueError("Require 2 <= batch size <= num_candidate <= 256")
    if num_candidate % num_batch_crystal:
        raise ValueError("This legacy ablation requires complete batches")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if initialization not in {
        "random-random",
        "random-AtomicMask",
        "Struct-random",
        "Struct-AtomicMask",
    }:
        raise ValueError(f"Unknown initialization: {initialization}")
    device = os.environ.get("SMOACS_DEVICE", "cuda")
    result_dir = str(Path(output_root or REPO_ROOT / "results").resolve() / exp_name)
    Path(result_dir).mkdir(parents=True, exist_ok=False)
    summary_path = os.path.join(result_dir, "result_summary.csv")
    input_dir = SECTION_DIR / "templates_masks" / "initial_structures"
    run_config = {
        "initialization": initialization,
        "method": method,
        "model_name": model_name,
        "device": device,
        "num_candidate": num_candidate,
        "b_size": num_batch_crystal,
        "num_steps": num_steps,
        "seed": seed,
        "exp_name": exp_name,
        "output_root": str(Path(result_dir).parent),
        "target_bandgap": target_bandgap,
        "bandgap_margin": bandgap_margin,
        "E_form_criteria": E_form_criteria,
        "atom_lr": atom_lr,
        "lattice_lr": lattice_lr,
        "coords_lr": coords_lr,
        "atom_cycle": atom_cycle,
        "lattice_cycle": lattice_cycle,
        "coords_cycle": coords_cycle,
        "angle_range": angle_range,
        "abc_range": abc_range,
        "optimize_mode": optimize_mode,
        "input_dir": str(input_dir),
        "template_ids": (input_dir / "template_order.txt")
        .read_text()
        .splitlines()[:num_candidate],
        "frozen_masks": str(
            SECTION_DIR / "templates_masks" / "masks" / "initial_masks.npz"
        ),
        "model_paths": [
            str(REPO_ROOT / "models" / "crystalformer" / name)
            for name in (
                "default.json",
                "megnet-bandgap-layer4.ckpt",
                "megnet-e_form-layer4.ckpt",
            )
        ],
    }
    (Path(result_dir) / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ----------------- Hyperparameters ----------------- #

    mask_method = "super_common"  # 'common' 推奨？　酸化数を現実的なものに制限できる
    tolerance_range = (None, None)  # ペロブスカイトのトレランス
    use_atomic_mask = True  # 基データの電荷に従った原子マスクを使うかどうか
    use_formation_energy = True  # formation energyの最適化を行うかどうか
    adding_noise_scale = None  # None or float
    e_form_min = None  # None or float
    angle_optimization = True
    mode = "band"  # 'band', 'min', 'max'
    acceptable_margin = 1.0

    # 引数設定の反映
    if target_bandgap is None:
        target_bandgap_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    elif type(target_bandgap) == float:
        target_bandgap_list = [target_bandgap]
    else:
        raise ValueError("target_bandgap must be float or None")
    print(f"<info> target_bandgap_list: {target_bandgap_list}")

    if trial is not None:
        # 学習率の設定
        atom_lr = float(
            f"{np.power(10, trial.suggest_float('atom_lr_pow', -5.0, 1.0)):.0e}"
        )
        lattice_lr = float(
            f"{np.power(10, trial.suggest_float('lattice_lr_pow', -5.0, 1.0)):.0e}"
        )
        coords_lr = float(
            f"{np.power(10, trial.suggest_float('coords_lr_pow', -5.0, 1.0)):.0e}"
        )
        print(f"<info> Atom. lr: {atom_lr}, cycle: {atom_cycle}")
        print(f"<info> Lattice. lr: {lattice_lr}, cycle: {lattice_cycle}")
        print(f"<info> Coords. lr: {coords_lr}, cycle: {coords_cycle}")

        # 良い解候補の選定設定
        cm_str_list = [
            "None",
            "50-100-150",
            "50-100",
            "100-150",
            "50",
            "100",
            "150",
            "60-120",
            "120",
        ]
        cm_str = trial.suggest_categorical("cm_str", cm_str_list)
        mutation_noise = float(
            f"{np.power(10, trial.suggest_float('mutation_noise_pow', -6.0, -0.6)):.0e}"
        )
        mutation_noise = f"{mutation_noise:.10f}".rstrip("0").rstrip(".")

        if perovskite_size in ["2x2x2", "3x3x3", "4x4x4"]:
            perovskite_size_dict = {"2x2x2": 2, "3x3x3": 3, "4x4x4": 4}
            mutation_noise = (
                f"{(float(mutation_noise) / perovskite_size_dict[perovskite_size]):.0e}"
            )
            mutation_noise = f"{float(mutation_noise):.10f}".rstrip("0").rstrip(".")
        top_ratio = np.round(trial.suggest_float("top_ratio", 0.0, 1.0), 2)
        group_C_use_rate = np.round(
            trial.suggest_float("group_C_use_rate", 0.0, 1.0), 2
        )
        atom_dist_init = trial.suggest_categorical("atom_dist_init", [True, False])
        print(
            f"<info> cm_str: {cm_str}, mutation_noise: {mutation_noise}, top_ratio: {top_ratio}, group_C_use_rate: {group_C_use_rate}, atom_dist_init: {atom_dist_init}"
        )
        opt_setting_name = f"pytorch_optimizer___steps_cm-{cm_str}__mutation_noise-{mutation_noise}__top_ratio-{top_ratio}__group_C_use_rate-{group_C_use_rate}__atom_dist_init-{atom_dist_init}"
        assert parse_evolution_parameters(opt_setting_name)
        optimize_mode = opt_setting_name

        ## 以下はALIGNNのみ
        if model_name == "ALIGNN":
            num_update = trial.suggest_int("num_update", 1, 50)
            print(f"<info> ALIGNN num_update: {num_update}")
            num_update = trial.suggest_int("num_update", 1, 50)
            ALIGNN_update_step = calculate_update_steps(
                num_steps=num_steps, num_update=num_update
            )

        assert (
            all_reusult_dict is not None
        ), "if trial is not None, all_results must be provided."

    else:
        all_reusult_dict = []

    # super cell 最適化の修正
    if perovskite_size in ["2x2x2", "3x3x3", "4x4x4"]:
        print(
            f"<info> modify coords_lr for larger perovskite {perovskite_size}. coords_lr:",
            coords_lr,
            end=" -> ",
        )
        perovskite_size_dict = {"2x2x2": 2, "3x3x3": 3, "4x4x4": 4}
        coords_lr = coords_lr / perovskite_size_dict[perovskite_size]
        coords_lr = float(str(round(coords_lr, 6)))
        print(
            f"{coords_lr} \n<info> modify limit_coords_displacement for larger perovskite {limit_coords_displacement}. limit_coords_displacement:",
            limit_coords_displacement,
            end=" -> ",
        )
        limit_coords_displacement = (
            limit_coords_displacement / perovskite_size_dict[perovskite_size]
        )
        limit_coords_displacement = float(str(round(limit_coords_displacement, 6)))
        print(
            f"{limit_coords_displacement} \n<info> modify abc_range for larger perovskite {abc_range}. abc_range:",
            abc_range,
            end=" -> ",
        )
        abc_range = np.array(abc_range) * perovskite_size_dict[perovskite_size]
        abc_range = tuple(abc_range)
        print(f"{abc_range}")

    # del  use_formation_energy, num_steps
    # for use_formation_energy in [False, True]:
    #    for num_steps in [50, 100, 200]:

    # ------------------------------------------------- #
    # ------------------------------------------------- #
    # -----------------  Main Loop   ------------------ #
    # --------------------------------------------------#
    # ------------------------------------------------- #
    num_experiment = 0
    # for atom_lr, lattice_lr, coords_lr, atom_cycle, lattice_cycle, coords_cycle in itertools.product(atom_lr_list, lattice_lr_list, coords_lr_list, atom_cycle_list, lattice_cycle_list, coords_cycle_list):
    # for crystal_system, perovskite_mode in zip(['perovskite', None],['random', None]):
    # for perovskite_size in ['3x3x3']:#,'2x2x2', '3x3x3']:
    for _ in range(1):  # dummy loop for indent
        # dataset
        dir_path = str(SECTION_DIR / "templates_masks" / "initial_structures")
        dataset_name = "megnet_max_10atoms_neutral_super_common"
        intermid_dir_path = str(SECTION_DIR / "templates_masks" / "masks")

        if test_mode:
            num_candidate = 8  # len(os.listdir(dir_path))
            num_batch_crystal = 8  # 1024
            num_steps = 200
            limit_num = 1
            test_noise_scales = []
        else:
            limit_num = 1000

        if model_name == "Crystalformer":
            optimize_solution = optimize_solution_Crystalformer
            evaluation_for_each_batch = evaluation_for_each_batch_Crystalformer
            load_pretrained = Load_Pretrained_Crystalformers
            num_display = 12
        elif model_name == "ALIGNN":
            optimize_solution = optimize_solution_ALIGNN
            evaluation_for_each_batch = evaluation_for_each_batch_ALIGNN
            load_pretrained = Load_Pretrained_ALIGNN
            num_display = 12
        elif model_name == "Both":
            optimize_solution = optimize_solution_Both
            evaluation_for_each_batch = evaluation_for_each_batch_Both
            load_pretrained = Load_Pretrained_Both
            num_display = 6
        else:
            raise NotImplementedError

        # if model_name == 'ALIGNN' and test_mode and num_candidate != num_batch_crystal:
        #    raise ValueError("TEST_mode of ALIGNN model requires num_candidate == num_batch_crystal")

        num_atom_type = 98
        num_display = 6
        time_series_save = True

        # ----------------- Perovskite settings ----------------- #
        if crystal_system == "perovskite":
            tolerance_range = (0.8, 1.0)
            angle_optimization = False
            tolerance_loss_func = toleranceLoss(
                low=tolerance_range[0], high=tolerance_range[1]
            )
            max_ox_states = 2
        else:
            tolerance_loss_func = None
            angle_optimization = True
            max_ox_states = 10
            if limit_coords_displacement is not None:
                raise ValueError(
                    "limit_coords_displacement must be None for non-perovskite system"
                )

        # 損失やソフトマックス温度のスケジュールの設定
        dist_temp_sche, Ef_coef_schedule, step_of_x_opt_stop = set_schedule(
            num_steps=num_steps, ef_coef_base=float(use_formation_energy)
        )

        # -----------------  load pretrained models ------------------ #
        if model_name == "Crystalformer":
            bandgap_model, e_form_model = load_pretrained(device=device)
        else:
            bandgap_model, e_form_model = load_pretrained()
        safe_clear_output()

        # create setting dict
        settings_dict = {
            "initialization": initialization,
            "initialization_generator": torch.Generator(device=device).manual_seed(
                seed
            ),
        }
        settings_dict["intermid_dir_path"] = intermid_dir_path
        settings_dict["crystal_system"] = crystal_system
        settings_dict["max_ox_states"] = max_ox_states
        settings_dict["num_candidate"] = num_candidate
        settings_dict["num_batch_crystal"] = num_batch_crystal
        settings_dict["use_atomic_mask"] = use_atomic_mask
        settings_dict["mask_method"] = mask_method
        settings_dict["test_mode"] = test_mode
        settings_dict["learning_rates"] = [lattice_lr, atom_lr, coords_lr]
        settings_dict["learning_rate_cycle"] = [lattice_cycle, atom_cycle, coords_cycle]
        settings_dict["num_steps"] = num_steps
        settings_dict["angle_optimization"] = angle_optimization
        settings_dict["test_mode"] = test_mode
        settings_dict["angle_range"] = angle_range
        settings_dict["abc_range"] = abc_range
        settings_dict["optimize_mode"] = optimize_mode

        if ALIGNN_update_step is not None:
            settings_dict["ALIGNN_update_step"] = (
                ALIGNN_update_step  # list(np.arange(200)+1)
            )
        else:
            settings_dict["ALIGNN_update_step"] = []

        # return settings_dict
        print(tolerance_loss_func)

        # --------------------------------------------------------- #
        # -----------------  band gap Loop   ------------------ #
        # ----------------------------------------------------------#
        for target_bandgap in target_bandgap_list[:limit_num]:
            gap_min = target_bandgap - bandgap_margin
            gap_max = target_bandgap + bandgap_margin
            gap_loss_func = GapLoss(gap_min=gap_min, gap_max=gap_max, mode=mode)
            # ----------------- Experiment and File Names ----------------- #
            configuration_name = define_experiment_name(
                model_name=model_name,
                dataset_name=dataset_name,
                optimize_mode=optimize_mode,
                target_bandgap=target_bandgap,
                bandgap_margin=bandgap_margin,
                use_formation_energy=use_formation_energy,
                e_form_min=e_form_min,
                adding_noise_scale=adding_noise_scale,
                learning_rates=[lattice_lr, atom_lr, coords_lr],
                learning_cycles=[lattice_cycle, atom_cycle, coords_cycle],
                num_steps=num_steps,
                num_graph_update=len(settings_dict["ALIGNN_update_step"]),
            )
            print(exp_name)
            print(extract_conditions(configuration_name))
            npz_path = os.path.join(result_dir, "result.npz")
            csv_path = os.path.join(result_dir, "result.csv")
            poscar_saved_dir = os.path.join(result_dir, "poscar")
            os.makedirs(poscar_saved_dir, exist_ok=False)
            history_img_path = os.path.join(result_dir, "history_img.png")

            # -----------------  Load Data ------------------ #
            all_inputs_dict = load_initial_data(
                device=device,
                settings_dict=settings_dict,
                model_name=model_name,
                dir_path=dir_path,
                intermid_dir_path=settings_dict["intermid_dir_path"],
                crystal_system=settings_dict["crystal_system"],
                max_ox_states=settings_dict["max_ox_states"],
                num_candidate=settings_dict["num_candidate"],
                batch_size=settings_dict["num_batch_crystal"],
                use_atomic_mask=settings_dict["use_atomic_mask"],
                mask_method=settings_dict["mask_method"],
                test_mode=settings_dict["test_mode"],
                angle_range=settings_dict["angle_range"],
                abc_range=settings_dict["abc_range"],
            )
            # -----------------  Optimization ------------------ #
            optimized_dict_list, time_series_list = [], []
            for mini_batch_data in tqdm(
                all_inputs_dict, desc="mini_batch", leave=False
            ):
                # select batch candidate crystals
                mini_batch_inputs_dict, scalers = mini_batch_data

                # Optimization
                optimized_mini_batch_inputs_dict, time_series, scalers = (
                    optimize_solution(
                        settings_dict=settings_dict,
                        mini_batch_inputs_dict=mini_batch_inputs_dict,
                        learning_rates=[lattice_lr, atom_lr, coords_lr],
                        learning_rate_cycle=[lattice_cycle, atom_cycle, coords_cycle],
                        num_steps=num_steps,
                        dist_temp_sche=dist_temp_sche,
                        Ef_coef_schedule=Ef_coef_schedule,
                        step_of_x_opt_stop=step_of_x_opt_stop,
                        gap_loss_func=gap_loss_func,
                        tolerance_loss_func=tolerance_loss_func,
                        angle_optimization=angle_optimization,
                        bandgap_model=bandgap_model,
                        e_form_model=e_form_model,
                        adding_noise_scale=adding_noise_scale,
                        scalers=scalers,
                        crystal_system=crystal_system,
                        limit_coords_displacement=limit_coords_displacement,
                        device=device,
                        time_series_save=time_series_save,
                        e_form_min=e_form_min,
                    )
                )
                # Evaluation
                optimized_mini_batch_inputs_dict = evaluation_for_each_batch(
                    settings_dict=settings_dict,
                    optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
                    scalers=scalers,
                    bandgap_model=bandgap_model,
                    e_form_model=e_form_model,
                    Ef_coef_schedule=Ef_coef_schedule,
                    gap_loss_func=gap_loss_func,
                    crystal_system=crystal_system,
                    limit_coords_displacement=limit_coords_displacement,
                    tolerance_loss_func=tolerance_loss_func,
                    adding_noise_scale=None,
                    e_form_min=e_form_min,
                )
                # noise が加えられた場合の評価
                for noise_scale in test_noise_scales:
                    optimized_mini_batch_inputs_dict = noise_robust_test(
                        optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
                        noise_scale=noise_scale,
                        evaluation_for_each_batch=evaluation_for_each_batch,
                        scalers=scalers,
                        bandgap_model=bandgap_model,
                        e_form_model=e_form_model,
                        Ef_coef_schedule=Ef_coef_schedule,
                        gap_loss_func=gap_loss_func,
                        crystal_system=crystal_system,
                        tolerance_loss_func=tolerance_loss_func,
                        e_form_min=e_form_min,
                    )

                optimized_dict_list.append(optimized_mini_batch_inputs_dict)

                time_series_list.append(time_series)

            # -----------------  Save and Visualization ------------------ #
            save_optimization_results(
                npz_path=npz_path,
                optimized_dict_list=optimized_dict_list,
                model_name=model_name,
                crystal_system=crystal_system,
                test_noise_scales=test_noise_scales,
            )
            # fin_poscar_dir = os.path.join(result_dir, 'poscar')
            # os.makedirs(fin_poscar_dir, exist_ok=True)
            # save_poscar(optimized_mini_batch_inputs_dict, fin_poscar_dir)
            # safe_clear_output()

            d = np.load(npz_path, allow_pickle=True)
            abc_strings, angle_strings = [], []
            for abc in d["batch_abc"]:
                abc_strings.append(f"[{abc[0]:.3f}, {abc[1]:.3f}, {abc[2]:.3f}]")
            for angle in d["batch_angle"]:
                angle_strings.append(
                    f"[{angle[0]:.3f}, {angle[1]:.3f}, {angle[2]:.3f}]"
                )
            df = pd.DataFrame(
                {
                    "lattice_index": np.arange(len(d["size"])),
                    "bandgap_onehot": d["bandgap_onehot"],
                    "ef_onehot": d["ef_onehot"],
                    "loss_onehot": d["loss_onehot"],
                    "num_atoms": d["num_atoms"],
                    "original_fnames": d["original_fnames"],
                    "tolerance": d["tolerance"],
                    "batch_abc": abc_strings,
                    "batch_angle": angle_strings,
                }
            ).sort_values(["loss_onehot", "ef_onehot"])
            df.reset_index(inplace=True)
            df.set_index("lattice_index", inplace=True)
            df.to_csv(csv_path)
            print(len(d["lattice_vectors"]), num_candidate)
            assert len(d["lattice_vectors"]) == num_candidate
            # assert len(d['dir_coords']) == len(d['batch'])

            # 最適化された構造を保存
            df2 = check_neurality_bondlength_and_save_structure(
                npz_path,
                csv_path,
                poscar_saved_dir,
                crystal_system,
                perovskite_size,
                neutral_check=True,
                acceptable_margin=acceptable_margin,
            )
            safe_clear_output()

            # 最適化の様子を可視化する
            print(exp_name)
            optimization_history_display_for_promising_candidates(
                gap_min,
                gap_max,
                npz_path,
                csv_path,
                model_name,
                num_display=20,
                history_img_path=history_img_path,
            )

            # E_formの判定
            df["Ef_success"] = df["ef_onehot"] < E_form_criteria

            try:
                df = pd.concat(
                    [
                        df.reset_index().set_index("original_fnames"),
                        df2.set_index("original_fnames"),
                    ],
                    axis=1,
                )
            except KeyError:
                # すべてがNaNの時. からのDataFrameを作成し、保存する
                df = pd.DataFrame([])
                df.to_csv(csv_path)
                # ハイパラ調整時はNaNを返す
                if trial is not None:
                    return 0.0
                continue

            # valid structureの判定
            if crystal_system == "perovskite" and perovskite_size in [
                "2x2x2",
                "3x3x3",
                "4x4x4",
            ]:
                # 原子数が多すぎて中性を計算できないので、valid_sturctureから中性の判定を除く
                df["valid_structure"] = df["minbond_less_than_0.5"]
            else:
                df["valid_structure"] = (
                    df["is_neutral_cmn"] & df["minbond_less_than_0.5"]
                )

            # ペロブスカイト専用の判定
            if crystal_system == "perovskite":
                tolerence_bool = (min(tolerance_range) <= df["tolerance"]) & (
                    df["tolerance"] <= max(tolerance_range)
                )
                df["tolerance_success"] = tolerence_bool
                df["perov_coords"] = perovskite_coordinate_check(
                    d["init_coords"],
                    d["dir_coords"],
                    d["size"],
                    limit_of_displacement=limit_coords_displacement + 1e-6,
                )
                df["perov_success"] = (tolerence_bool) & df["perov_coords"]
            else:
                df["tolerance_success"] = [np.nan] * df.shape[0]
                df["tolerance"] = [np.nan] * df.shape[0]
                df["perov_coords"] = [np.nan] * df.shape[0]
                df["perov_success"] = [np.nan] * df.shape[0]

            # 全体の成功の判定
            df["bandgap_success"] = (
                df["loss_onehot"] == 0
            )  # bandgapのlossが0のものが成功とする
            if crystal_system == "perovskite":
                df["success"] = (
                    df["valid_structure"]
                    & df["bandgap_success"]
                    & df["perov_success"]
                    & df["Ef_success"]
                )
            else:
                df["success"] = (
                    df["valid_structure"] & df["bandgap_success"] & df["Ef_success"]
                )

            print(
                f"success rate: {df.success.sum() / df.shape[0]:.3f} ({df.success.sum()}/{df.shape[0]})"
            )
            df.to_csv(csv_path)

            safe_display(df.head(50))
            ## 最適化されていく様子のGIFファイルを作成する
            # lattice_mat_history = np.concatenate([np.stack(time_series[0],axis=0) for time_series in time_series_list],axis=1) # shape = (num_steps, num_candidate, 3, 3)
            # coords_history = np.concatenate([np.stack(time_series[1],axis=0) for time_series in time_series_list],axis=1) # shape = (num_steps, num_candidate, num_atoms, 3)
            # normalized_coords_history = np.concatenate([np.stack(time_series[2],axis=0) for time_series in time_series_list],axis=1) # shape = (num_steps, num_candidate, num_atoms, 3)
            # abc_history = np.concatenate([np.stack(time_series[3],axis=0) for time_series in time_series_list],axis=1) # shape = (num_steps, num_candidate, 3)
            # angle_history = np.concatenate([np.stack(time_series[4],axis=0) for time_series in time_series_list],axis=1) # shape = (num_steps, num_candidate, 3)
            # elememts_history = np.concatenate([np.stack(time_series[5],axis=0) for time_series in time_series_list],axis=1) # shape = (num_steps, num_candidate, num_atoms)
            # np.savez(
            #    os.path.join(saved_gif_dir, 'time_series.npz'),
            #    lattice_mat_history=lattice_mat_history,
            #    coords_history=coords_history,
            #    normalized_coords_history=normalized_coords_history,
            #    abc_history=abc_history,
            #    angle_history=angle_history,
            #    elememts_history=elememts_history,
            # )

            # all_time_series = [lattice_mat_history, coords_history, abc_history, angle_history, elememts_history]
            # create_optimization_GIF_file(all_time_series, gap_min, gap_max, npz_path, csv_path, num_steps, saved_gif_dir, tmp_output_dir, num_saves=20, save_time_step=5)
            def list2str(elements):
                if type(elements) == list:
                    elements.sort()
                return str(elements)

            unique_elem_list = df["elements"].apply(list2str).unique().tolist()
            if "nan" in unique_elem_list:
                unique_elem_list.remove("nan")
            success_unique_elem_list = (
                df.loc[df["success"] == 1, "elements"].apply(list2str).unique().tolist()
            )
            additional_result_dict = {
                "num_steps": num_steps,
                "element_diversity": len(unique_elem_list) / df.shape[0],
                "success_element_diversity": np.divide(
                    len(success_unique_elem_list), df.loc[df["success"] == 1].shape[0]
                ),
            }

            result_dict = (
                df[
                    [
                        "success",
                        "bandgap_success",
                        "Ef_success",
                        "valid_structure",
                        "minbond_less_than_0.5",
                        "is_neutral_cmn",
                        "perov_success",
                        "perov_coords",
                        "tolerance_success",
                        "tolerance",
                    ]
                ]
                .mean()
                .to_dict()
            )
            result_dict = (
                extract_conditions(configuration_name)
                | {
                    "initialization": initialization,
                    "seed": seed,
                    "num_candidate": num_candidate,
                    "batch_size": num_batch_crystal,
                }
                | result_dict
                | additional_result_dict
            )
            all_reusult_dict.append(result_dict)
            pd.DataFrame(all_reusult_dict).to_csv(summary_path)
            num_experiment += 1

    safe_clear_output()
    print(exp_name)
    safe_display(history_img_path)
    os.makedirs(result_dir, exist_ok=True)
    pd.DataFrame(all_reusult_dict).to_csv(summary_path)
    # pd.DataFrame(all_reusult_dict).to_csv(f"{result_dir}.csv")
    safe_display(pd.DataFrame(all_reusult_dict))
    safe_display(pd.DataFrame(df))
    print("npz_path:", npz_path)
    try:
        success_rate = pd.read_csv(summary_path)["success"].values[0]
        success_diversity = pd.read_csv(summary_path)[
            "success_element_diversity"
        ].values[0]
        # print("time_serices saved:", os.path.join(saved_gif_dir, 'time_series.npz'))
        print("total success rate:", success_rate)
        print("success element diversity:", success_diversity)
    except KeyError:
        success_rate = 0.0
        success_diversity = 0.0
        print("No success rate or diversity")

    if trial is not None:
        assert num_experiment == 1
        return success_rate * success_diversity

    return result_dir


INITIALIZATIONS = (
    "random-random",
    "random-AtomicMask",
    "Struct-random",
    "Struct-AtomicMask",
)
METHODS = ("grad", "grad_sa", "mutation", "mutation_sa")


def main():
    parser = argparse.ArgumentParser(
        description="Section 2.1 Crystalformer ablation study"
    )
    parser.add_argument("--initialization", choices=INITIALIZATIONS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--num_candidate", type=int, default=128)
    parser.add_argument("--b_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument(
        "--mutation_steps",
        type=int,
        nargs="+",
        default=[100, 150],
        help="Zero-based mutation step indices (default: 100 150)",
    )
    parser.add_argument("--output_root", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    if args.method.startswith("mutation"):
        if any(step < 0 or step >= args.num_steps for step in args.mutation_steps):
            parser.error("mutation_steps must be zero-based indices below num_steps")
        steps = "-".join(str(step) for step in args.mutation_steps)
        optimize_mode = (
            f"pytorch_optimizer___steps_cm-{steps}__mutation_noise-0.00007"
            "__top_ratio-0.10__group_C_use_rate-0.05__atom_dist_init-True"
        )
    else:
        optimize_mode = "pytorch_optimizer___steps_cm-None"
    if args.method.endswith("_sa"):
        optimize_mode += "___sa___mode-exponential__alpha-0.9__T0-0.1__noise_scale-0.1"
    main_experiment(
        model_name="Crystalformer",
        optimize_mode=optimize_mode,
        num_steps=args.num_steps,
        num_candidate=args.num_candidate,
        num_batch_crystal=args.b_size,
        bandgap_margin=0.04,
        target_bandgap=2.5,
        crystal_system=None,
        perovskite_size=None,
        limit_coords_displacement=None,
        atom_lr=0.007,
        lattice_lr=0.002,
        coords_lr=0.005,
        test_noise_scales=[],
        initialization=args.initialization,
        method=args.method,
        seed=args.seed,
        exp_name=args.exp_name,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
