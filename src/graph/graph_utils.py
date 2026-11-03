import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes
from torch.utils.data import DataLoader

from third_party.alignn_minimal.graphs import Graph, StructureDataset
from src.utils.common import create_element_list
from src.utils.coord import direct_to_cartesian_batch
from tqdm import tqdm

from src.utils.coord import compute_abc_angle, compute_lattice_vectors, calculate_bond_cosine
from src.utils.mask_utils import create_learnable_oxides_mask
from src.spg_wp.wyckoff_expander import WyckoffExpander, assert_wyckoff_expansion_matches_atoms


def load_graphs(
    dataset=[],
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    cutoff_extra: float = 3,
    max_neighbors: int = 12,
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

    # print('dataset',dataset,type(dataset))
    print("Converting to graphs!")
    graphs = []
    # columns=dataset.columns
    wyckoff_expansion_usages = torch.tensor([False]*len(dataset))

    for row_i, row in tqdm(dataset.iterrows(),total=len(dataset)):
        # print('iooooo',i)
        
        atoms = row["atoms"]
        atom_mask_for_all_ox_states = row["atom_mask_for_all_ox_states"]
        ox_states_used_mask = row["ox_states_used_mask"]
        radii_for_all_ox_states = row["radii_for_all_ox_states"]
        site_ids = row["site_ids"]
        id_tag = row.get(id_tag, None)

        # Wyckoff expansionを使う場合は、ノード数とテンソルの形状が一致しないため、zero-paddingを行う
        if len(row['elements']) != len(site_ids):
            if row_i == 0:
                print("Wyckoff expansion is used, but the number of elements and site_ids do not match. Zero-padding will be applied.")
            wyckoff_expansion_usages = torch.tensor([True]*len(row['elements']))

            # padding
            pad_size = len(row['elements']) - len(site_ids)
            atom_mask_for_all_ox_states = np.concatenate([atom_mask_for_all_ox_states, np.nan*np.zeros((pad_size, atom_mask_for_all_ox_states.shape[1],atom_mask_for_all_ox_states.shape[2]))], axis=0)
            radii_for_all_ox_states = np.concatenate([radii_for_all_ox_states, np.nan*np.zeros((pad_size, radii_for_all_ox_states.shape[1], radii_for_all_ox_states.shape[2]))], axis=0)
            site_ids = np.concatenate([site_ids, np.nan*np.zeros(pad_size)], axis=0)

        # Wyckoff positionの数を取得 (Wyckoff positionの数は、site_idsのの数). Wyckoff position拡張を使わない場合は、site_idsの数をいれる
        WP_size = [len(site_ids)]*len(site_ids)


        # Fallback geometry must already be reflected in the shared dataset.
        geometry = atoms if isinstance(atoms, dict) else atoms.to_dict()
        if not all(
            np.isfinite(geometry[key]).all()
            for key in ("lattice_mat", "coords", "abc", "angles")
        ):
            raise ValueError(
                f"{row['jid']}: nonfinite geometry before graph construction"
            )

        structure = (
            Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
        )
        g = Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            cutoff_extra=cutoff_extra,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
            neighbor_strategy=neighbor_strategy,
            id=id_tag,
        )

        #g.ndata["atom_mask_for_all_ox_states"] = torch.tensor(
        #    atom_mask_for_all_ox_states
        #).type(torch.get_default_dtype())
        #g.ndata["ox_states_used_mask"] = torch.tensor(
        #    torch.stack([torch.tensor(ox_states_used_mask)]*len(atom_mask_for_all_ox_states))
        #).type(torch.get_default_dtype())
        #g.ndata["radii_for_all_ox_states"] = torch.tensor(
        #    radii_for_all_ox_states
        #).type(torch.get_default_dtype())
        if not (g.num_nodes() == len(atom_mask_for_all_ox_states) == len(site_ids)):
            raise ValueError(
                f"{row['jid']}: graph/mask/site_ids mismatch: "
                f"nodes={g.num_nodes()}, mask={len(atom_mask_for_all_ox_states)}, "
                f"site_ids={len(site_ids)}"
            )
        g.ndata["crystal_id"] = torch.tensor([row_i]*len(atom_mask_for_all_ox_states), dtype=torch.int64)
        g.ndata["site_ids"] = torch.tensor(site_ids).long()

        #g.ndata['use_Wyckoff_expansion'] = wyckoff_expansion_usages
        #g.ndata['WP_size'] = torch.tensor(WP_size)
        graphs.append(g)


    return graphs


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

    #print("StructureDatasetに入力する変数")
    #print("target=",target)
    #print("target_atomwise=",target_atomwise, type(target_atomwise))
    #print("target_grad=",target_grad, type(target_grad))
    #print("target_stress=",target_stress, type(target_stress))
    #print("atom_features=",atom_features, type(atom_features))
    #print("line_graph=",line_graph, type(line_graph))
    #print("id_tag=",id_tag)
    #print("classification=",classification)


    st_data = StructureDataset(
        df[['jid','atoms','dummy_target']],
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


def get_all_atoms_coords(
    train_df: pd.DataFrame,
    wyckoff_expander: WyckoffExpander
) -> Tuple[
    torch.Tensor,  # batch_dir_coords
    torch.Tensor,  # batch_dst_ids
    torch.Tensor,  # batch_src_ids
    torch.Tensor,  # batch_displace
    torch.Tensor,  # batch_lattice_vectors
    List[int],  # num_atoms
    List[int],  # num_edges
]:
    """
    Convert atomic structures in a DataFrame to batched coordinate and edge tensors for graph processing.

    Args:
        train_df (pd.DataFrame): DataFrame containing a column `atoms` (dict or Atoms) for each crystal structure.
        wyckoff_expander (wyckoff_expander): An object that provides the method to expand Wyckoff positions.

    Returns:
        Tuple[
            torch.Tensor: Batched direct coordinates of all atoms (shape: [total_atoms, 3]).
            torch.Tensor: Destination atom indices of all edges (shape: [total_edges]).
            torch.Tensor: Source atom indices of all edges (shape: [total_edges]).
            torch.Tensor: Displacement vectors between atoms in fractional coordinates (shape: [total_edges, 3]).
            torch.Tensor: Lattice vectors for each edge (shape: [total_edges, 3, 3]).
            List[int]: Number of atoms per structure.
            List[int]: Number of edges per structure.
        ]
    """
    _dir_coords, _lattice_vectors = [], []
    _dst_ids, _src_ids, _displace = [], [], []
    num_atoms, num_edges = [], []
    num_total_atoms = 0
    site_ids_list, ox_states_used_mask_list, radii_for_all_ox_states_list = [], [], []
    atom_mask_for_all_ox_states_list = []
    wp_size_list = []

    for _, row in train_df.iterrows():
        atoms_dict = row['atoms']
        wp_coords = row.get('wp_coords', None)
        site_ids = row['site_ids']
        ox_states_used_mask = row['ox_states_used_mask']
        atom_mask_for_all_ox_states = row['atom_mask_for_all_ox_states']
        radii_for_all_ox_states = row['radii_for_all_ox_states']

        atoms = Atoms.from_dict(atoms_dict)
        if wp_coords is not None:
            # if wp_coords is not None, use Wyckoff expansion and use wp_coords
            try: # debug
                assert_wyckoff_expansion_matches_atoms(
                    atoms=atoms,
                    wp_coords=wp_coords,
                    site_ids=site_ids,
                    wyckoff_expander=wyckoff_expander
                )
                _dir_coords.append(torch.tensor(wp_coords))
            except AssertionError as e:
                expanded_coords, _ = wyckoff_expander.wyckoff_expand(
                    wp_coords=torch.tensor(wp_coords).to(wyckoff_expander.device),
                    wp_atom_dist=torch.zeros_like(torch.tensor(wp_coords)).to(wyckoff_expander.device),
                    site_ids=torch.tensor(site_ids).to(wyckoff_expander.device),
                )
                print(f"AssertionError: {e}")
                print("Using original fractional coordinates instead.")
                print("frac_coords = torch.", torch.tensor(np.array(atoms.frac_coords)))
                print("test_coords = torch.", torch.tensor(row['test_coords']))
                print("expanded_coords = torch.", expanded_coords)
                print("ex_coords = torch.",  row['ex_coords'])
                print("wp_coords = torch.", torch.tensor(wp_coords))
                print("site_ids:", site_ids)

                raise Exception("")
        else:
            _dir_coords.append(torch.Tensor(atoms.frac_coords))

        edges = nearest_neighbor_edges(atoms, use_canonize=True)
        y_src_ids, y_dst_ids, y_displace = build_undirected_edge_data_tensor(edges)
        lattice_vectors = torch.Tensor(atoms.lattice_mat)

        # Edges, lattice vectors, and coordinates
        _lattice_vectors.append(lattice_vectors.unsqueeze(0))
        _dst_ids.append(y_dst_ids + num_total_atoms)
        _src_ids.append(y_src_ids + num_total_atoms)
        _displace.append(y_displace)
        num_total_atoms += len(atoms.frac_coords)
        num_atoms.append(len(atoms.frac_coords))
        num_edges.append(len(y_dst_ids))
        # others
        site_ids_list.append(torch.tensor(site_ids, dtype=torch.int32))
        ox_states_used_mask_list.append(torch.stack([torch.tensor(ox_states_used_mask)] * len(site_ids), dim=0).float())
        atom_mask_for_all_ox_states_list.append(torch.tensor(atom_mask_for_all_ox_states, dtype=torch.float32))
        radii_for_all_ox_states_list.append(torch.tensor(radii_for_all_ox_states, dtype=torch.float32))
        wp_size_list.append(len(site_ids))

    batch_dir_coords = torch.cat(_dir_coords)
    batch_dst_ids = torch.cat(_dst_ids)
    batch_src_ids = torch.cat(_src_ids)
    batch_displace = torch.cat(_displace)
    batch_lattice_vectors = torch.cat(_lattice_vectors)
    site_ids = torch.cat(site_ids_list)
    ox_states_used_mask = torch.cat(ox_states_used_mask_list)
    atom_mask_for_all_ox_states = torch.cat(atom_mask_for_all_ox_states_list)
    radii_for_all_ox_states = torch.cat(radii_for_all_ox_states_list)
    wp_size = torch.tensor(wp_size_list, dtype=torch.int32)

    assert len(batch_dir_coords) == len(site_ids) == len(atom_mask_for_all_ox_states) == len(radii_for_all_ox_states) == len(ox_states_used_mask), f"Mismatch in lengths of batch_dir_coords ({batch_dir_coords.shape}), site_ids ({site_ids.shape}), atom_mask_for_all_ox_states ({atom_mask_for_all_ox_states.shape}), and radii_for_all_ox_states ({radii_for_all_ox_states.shape}) and ox_states_used_mask ({ox_states_used_mask.shape})"
    assert len(batch_lattice_vectors) == len(wp_size), "Mismatch in lengths of batch_lattice_vectors and ox_states_used_mask"

    return (
        batch_dir_coords,
        batch_dst_ids,
        batch_src_ids,
        batch_displace,
        batch_lattice_vectors,
        num_atoms,
        num_edges,
        site_ids,
        ox_states_used_mask,
        atom_mask_for_all_ox_states,
        radii_for_all_ox_states,
        wp_size
    )


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
        feats.append(get_node_attributes(elem, atom_features='cgcnn'))
    atom_feat_matrix = torch.Tensor(np.array(feats))
    return atom_feat_matrix


def graph_to_tensors(
        g:dgl.DGLGraph, 
        lg:dgl.DGLGraph, 
        train_df:pd.DataFrame, 
        wyckoff_expander: WyckoffExpander,
        use_atomic_mask:bool=False,
        )->Dict[str,torch.Tensor]:
    """
    Converts DGL graphs and corresponding atomic data into batched tensor dictionaries for training ALIGNN-style models.

    This function takes a crystal graph `g` and its corresponding line graph `lg`, along with metadata from a 
    structure dataframe `train_df`, and processes them into a dictionary of PyTorch tensors for training. 
    The resulting dictionary contains atomic coordinates, lattice parameters, atomic features, connectivity 
    information, and optionally learnable atomic masks and radii.

    Args:
        g (dgl.DGLGraph): The main DGL graph containing node and edge information for the crystal structure.
        lg (dgl.DGLGraph): The associated line graph used for capturing bond angle features.
        train_df (pd.DataFrame): A dataframe with one row per crystal, including atomic structure (`atoms`) 
                                 and elemental metadata.
        wyckoff_expander (wyckoff_expander): An instance of the Wyckoff position expander to handle Wyckoff positions.
        use_atomic_mask (bool, optional): If True, oxidation state masks and radii tensors are included. Default is False.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing all tensors needed for batched ALIGNN input, including:
            - 'batch_dir_coords': Direct coordinates of all atoms
            - 'batch_abc': Lattice lengths (a, b, c)
            - 'batch_angle': Lattice angles (α, β, γ)
            - 'init_coords', 'init_abc', 'init_angles': Clones of the above for initial values
            - 'atomic_distribution': Initialized uniform atomic distribution (learnable)
            - 'atom_feat_matrix': Element-wise feature matrix
            - 'batch_src_ids', 'batch_dst_ids': Source and destination atom indices for edges
            - 'z_src_ids', 'z_dst_ids': Source and destination IDs for line graph edges
            - 'batch_displace': Displacement vectors for periodic boundary conditions
            - 'num_edges': Number of edges per structure
            - 'site_ids': Site information (e.g., A/B/O for perovskites)
            - 'fnames': Structure identifiers (e.g., JID)
            - Optional keys when `use_atomic_mask=True`:
                - 'atomic_mask'
                - 'radii_tensor'
                - 'ox_mask_learnable_tensor_per_crystal'
    """


    # initialization for direct coordiantes, lattice vectors, src_ids, dst_ids, displace
    batch_dir_coords, batch_dst_ids, batch_src_ids, batch_displace, lattice_vectors, num_atoms, num_edges, site_ids, ox_states_used_mask, atom_mask_for_all_ox_states, radii_for_all_ox_states, wp_size = get_all_atoms_coords(train_df,wyckoff_expander)

    # Initialization of atomic distribution. 
    ### atom_feat_matrix.shape = (num_atom_species(98), num_features(92))
    atom_feat_matrix = generate_feature_matrix() 
    ### atomic_dist is initialized to a uniform distribution
    atomic_dist = torch.ones(len(site_ids),atom_feat_matrix.shape[0], dtype=torch.float32)
    atomic_dist = atomic_dist / atomic_dist.sum(1).view(-1,1)

    # initialization for lattice vectors: a,b,c,alpha,beta,gamma
    batch_abc, batch_angle = compute_abc_angle(lattice_vectors)

    # initialization for angles
    z_src_ids, z_dst_ids = lg.edges()

    # count the number of atoms in each crystal
    _size = []
    for crystal_id in g.ndata['crystal_id'].unique():
        _size.append(g.ndata['crystal_id'].eq(crystal_id).sum().item())
    size = torch.tensor(_size).long()


    ### to GPU 
    batch_dst_ids = batch_dst_ids#.to(device)
    batch_src_ids = batch_src_ids#.to(device)
    z_src_ids = z_src_ids#.to(device)
    z_dst_ids = z_dst_ids#.to(device)
    batch_displace = batch_displace#.to(device)
    num_edges = torch.tensor(num_edges)#.to(device)
    atom_feat_matrix = atom_feat_matrix#.to(device)

    # Wyckoff Positionsのサイズでない、結晶サイズのマスク. create_learnable_oxides_mask関数で利用する


    alignn_minibatch_dict = {
        'batch_dir_coords':batch_dir_coords,#.to(device),
        'batch_abc':batch_abc,#.to(device),
        'batch_angle':batch_angle,#.to(device),
        'init_coords':batch_dir_coords.clone(),#.to(device),
        'init_abc':batch_abc.clone(),#.to(device),
        'init_angles':batch_angle.clone(),#.to(device),
        'atomic_distribution':atomic_dist,#.to(device),
        'batch_dst_ids':batch_dst_ids,
        'batch_src_ids':batch_src_ids,
        'z_src_ids':z_src_ids,
        'z_dst_ids':z_dst_ids,
        'size':size,
        'site_ids':site_ids,
        'batch_displace':batch_displace,
        'num_edges':num_edges,
        'atom_feat_matrix':atom_feat_matrix,
        'g':g,
        'lg':lg,
        'ox_states_used_mask':ox_states_used_mask,#.to(device),
        'wp_size': wp_size, # used for create_learnable_oxides_mask func.
        #'original_elements':original_elements,
        
        'fnames':train_df['jid'].values,
        'atomic_mask':None,
        'radii_tensor':None,
        'ox_mask_learnable_tensor_per_crystal':None,
    }

    if use_atomic_mask:
        alignn_minibatch_dict['atomic_mask'] = atom_mask_for_all_ox_states#.to(device)
        ox_mask_learnable_tensor_per_crystal  = create_learnable_oxides_mask(alignn_minibatch_dict)
        alignn_minibatch_dict['ox_mask_learnable_tensor_per_crystal'] = ox_mask_learnable_tensor_per_crystal
        alignn_minibatch_dict['radii_tensor'] = radii_for_all_ox_states#.to(device)

    return alignn_minibatch_dict


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


def create_batch_lattice_vectors(lattice_vectors: torch.Tensor, num_edges: List[int]):
    """
    This function expands the lattice_vectors (num_candidates, 3, 3) for each crystal by the number of edges (bonds).
    Args:
        lattice_vectors (torch.Tensor): Lattice vectors of shape (num_candidates, 3, 3)
        num_edges (List[int]): List of number of edges for each crystal in the batch
    """
    batch_lattice_vectors = torch.cat(
        [
            torch.cat([lattice_vectors[i : i + 1]] * num)
            for i, num in enumerate(num_edges)
        ]
    )
    return batch_lattice_vectors


def compute_bondlength(
    coords: torch.Tensor,
    dst_ids: torch.Tensor,
    src_ids: torch.Tensor,
    displacement: torch.Tensor,
    lattice_vectors: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes bond vectors and bond lengths in Cartesian coordinates for a batch of crystal structures.

    This function calculates the vectors between bonded atom pairs by transforming their fractional coordinates
    (direct coordinates) into Cartesian space, while accounting for periodic boundary displacements. It returns
    both the 3D bond vectors and their corresponding scalar lengths.

    Args:
        coords (torch.Tensor): Fractional coordinates of all atoms in the batch. Shape: (N_atoms, 3).
        dst_ids (torch.Tensor): Indices of destination atoms for each bond. Shape: (N_edges,).
        src_ids (torch.Tensor): Indices of source atoms for each bond. Shape: (N_edges,).
        displacement (torch.Tensor): Periodic displacement vectors for each bond in fractional coordinates. Shape: (N_edges, 3).
        lattice_vectors (torch.Tensor): Lattice matrices for each bond. Shape: (N_edges, 3, 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - edge_vectors (torch.Tensor): Bond vectors in Cartesian coordinates. Shape: (N_edges, 3).
            - bondlength (torch.Tensor): Euclidean lengths of the bond vectors. Shape: (N_edges,).
    """

    edge_vectors = direct_to_cartesian_batch(
        coords[dst_ids] - coords[src_ids] - displacement, lattice_vectors
    )
    bondlength = torch.norm(edge_vectors, dim=1)

    return edge_vectors, bondlength


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
