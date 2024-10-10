# This file includes code from the National Institute of Standards and Technology (NIST).
# See the NIST license information in the LICENSE file.

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes
from tqdm import tqdm

from alignn.graphs import Graph, StructureDataset

from .create_oxidation_mask import create_learnable_oxides_mask
from .lattice_utils import compute_abc_angle, direct_to_cartesian_batch
from .utils import create_element_list


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
    ########################################
    [Note]This function is improved from the function in the following GitHub. It adds atom masks and site IDs.
    https://github.com/usnistgov/alignn/blob/main/alignn/dataset.py#L17
    ########################################

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
        structure = (
            Atoms.from_dict(atoms) if isinstance(atoms, dict) else atoms
        )
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
        graphs, _ = dgl.load_graphs(str(cachefile))
    else:
        print("Converting to graphs!")
        graphs = []
        # columns=dataset.columns
        for ii, i in tqdm(dataset.iterrows(),total=len(dataset)):
            
            atoms = i["atoms"]
            # aIn case atoms contains NaN, load the graph from init_poscar.
            if np.isnan(atoms['lattice_mat']).any() or np.isnan(atoms['coords']).any() or np.isnan(atoms['abc']).any() or np.isnan(atoms['angles']).any():
                atoms = Atoms.from_poscar(f'init_poscars/{i["jid"]}').to_dict()
                print(f'nan included in {i["jid"]}, so load from init_poscars to convert to graph')

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
                id=i[id_tag],
            )
            atom_mask_for_all_ox_states = i["atom_mask_for_all_ox_states"]
            ox_states_used_mask = i["ox_states_used_mask"]
            radii_for_all_ox_states = i["radii_for_all_ox_states"]
            g.ndata["atom_mask_for_all_ox_states"] = torch.tensor(
                atom_mask_for_all_ox_states
            ).type(torch.get_default_dtype())
            g.ndata["ox_states_used_mask"] = torch.tensor(
                torch.stack([ox_states_used_mask]*len(atom_mask_for_all_ox_states))
            ).type(torch.get_default_dtype())
            g.ndata["radii_for_all_ox_states"] = torch.tensor(
                radii_for_all_ox_states
            ).type(torch.get_default_dtype())
            g.ndata["crystal_id"] = torch.tensor([ii]*len(atom_mask_for_all_ox_states), dtype=torch.int64)
            g.ndata["site_ids"] = i["site_ids"]
            #print(i["atoms"]["elements"])
            #g.ndata['original_elements'] = torch.tensor(i["atoms"]["elements"])
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



    import pandas as pd

import os

import numpy as np
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
    """
    ########################################
    [Note]This function is improved from the function in the following GitHub. It adds atom masks and site IDs.
    https://github.com/usnistgov/alignn/blob/main/alignn/dataset.py#L108
    ########################################
    
    Get Torch Dataset.
    """
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


def get_all_atoms_coords(train_df:pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    '''
    Based on train_df, calculate the direct coordinate of all atoms and the src and dist of the bonds in the crystal structure data to be used.

    Args:
        train_df: Dataframe containing information of crystal structures to be used, which was created at the same time as creating the Pytorch Dataset
    Outputs:
        batch_dir_coords: tensor containing direct coordinates of all atoms
        batch_dst_ids: tensor containing the ID of the distination point of all bonds
        batch_src_ids: tensor containing the ID of the starting point of all bonds
        batch_displace: tensor containing the displacement of all bonds
        batch_lattice_vectors: tensor containing the lattice vectors of all crystal structures
    '''
    _dir_coords, _lattice_vectors = [], []
    _dst_ids, _src_ids, _displace = [], [], []
    num_atoms, num_edges = [], []
    num_total_atoms = 0
    for atoms_dict in train_df.atoms:
        atoms = Atoms.from_dict(atoms_dict)
        _dir_coords.append(torch.Tensor(atoms.frac_coords))
        edges =  nearest_neighbor_edges(atoms,use_canonize=True)
        y_src_ids, y_dst_ids, y_displace =  build_undirected_edge_data_tensor(edges)
        lattice_vectors = torch.Tensor(atoms.lattice_mat)
        #_lattice_vectors.append(torch.stack([lattice_vectors]*len(y_dst_ids)))
        _lattice_vectors.append(lattice_vectors.unsqueeze(0))
        _dst_ids.append(y_dst_ids+num_total_atoms)
        _src_ids.append(y_src_ids+num_total_atoms)
        _displace.append(y_displace)
        num_total_atoms += len(atoms.frac_coords)
        num_atoms.append(len(atoms.frac_coords))
        num_edges.append(len(y_dst_ids))

    batch_dir_coords = torch.cat(_dir_coords)
    batch_dst_ids = torch.cat(_dst_ids)
    batch_src_ids = torch.cat(_src_ids)
    batch_displace = torch.cat(_displace)
    batch_lattice_vectors = torch.cat(_lattice_vectors)
    
    return batch_dir_coords, batch_dst_ids, batch_src_ids, batch_displace, batch_lattice_vectors, num_atoms, num_edges


def generate_feature_matrix() -> torch.Tensor:
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







def create_batch_lattice_vectors(lattice_vectors:torch.Tensor, num_edges:List[int]):
    '''
    This function extends lattice_vectors (num_candidates, 3, 3) for each crystal structure by the number of edges (bonds).
    '''
    batch_lattice_vectors = torch.cat([torch.cat([lattice_vectors[i:i+1]]*num) for i, num in enumerate(num_edges)])
    return batch_lattice_vectors

def compute_bondlength(coords:torch.Tensor, dst_ids:torch.Tensor, src_ids:torch.Tensor, displacement:torch.Tensor, lattice_vectors:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    This function calculates the bond length of all bonds in the crystal structure data to be used based on the direct coordinates of all atoms and the lattice vectors of all crystal structures in the batch.
    Args:
        coords: ractional coordinates of all atoms in the crystal structure data to be used
        dst_ids: ID of the distination point of all bonds in the batch
        src_ids: ID of the starting point of all bonds in the batch
        displacement: displacement of all bonds in the batch
        lattice_vectors: lattice vectors. lattice vectors of all crystal structures in the batch

    Returns:
        edge_vectors: cartesian coordinates of all bonds in the batch
        bondlength: bond length of all bonds in the batch
    '''
                
    edge_vectors = direct_to_cartesian_batch(coords[dst_ids] - coords[src_ids] - displacement   , lattice_vectors)
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

    return {"h": bond_cosine}


def calculate_bond_cosine(r1:torch.Tensor, r2:torch.Tensor)->torch.Tensor:
    '''calculate cosine
    Args:
        r1: shape=(N,3)
        r2: shape=(N,3)
    '''
    bond_cosine = torch.nn.functional.cosine_similarity(r1, r2, dim=1)
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


def graph_to_tensors(
        g:dgl.DGLGraph, 
        lg:dgl.DGLGraph, 
        train_df:pd.DataFrame, 
        data_dict_for_test:Dict[str,torch.Tensor], 
        device:str='cuda', 
        use_atomic_mask:bool=False,
        test_mode:bool=False
        )->Dict[str,torch.Tensor]:
    """
    Graphデータから、学習に使用するtensorデータを生成する。

    """
    # initialize atomic featuresの初期化
    atom_feat_matrix = generate_feature_matrix() 

    # initialization. Generate learnable atomic distribution as a uniform distribution.
    atomic_dist = torch.ones(np.sum(train_df['elements'].apply(len).values),atom_feat_matrix.shape[0])
    atomic_dist = atomic_dist / atomic_dist.sum(1).view(-1,1)
    
    # initialization for atomic masks
    atom_mask_for_all_ox_states = g.ndata.pop('atom_mask_for_all_ox_states')
    ox_states_used_mask = g.ndata.pop('ox_states_used_mask')
    radii_for_all_ox_states = g.ndata.pop('radii_for_all_ox_states')
    site_ids = g.ndata.pop('site_ids')

    # initialization for direct coordiantes, lattice vectors, src_ids, dst_ids, displace
    batch_dir_coords, batch_dst_ids, batch_src_ids, batch_displace, lattice_vectors, num_atoms, num_edges = get_all_atoms_coords(train_df)

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
    #g = g.to(device)
    #lg = lg.to(device)
    

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



def nearest_neighbor_edges(
    atoms=None,
    cutoff=8,
    max_neighbors=12,
    id=None,
    use_canonize=False,
):
    """
    Construct k-NN edge list.
    #############################
    This code comes from the following GitHub code.
    https://github.com/usnistgov/alignn/blob/main/alignn/graphs.py#L55
    #############################

    """
    # returns List[List[Tuple[site, distance, index, image]]]
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    #print("len(all_neighbors):",len(all_neighbors))

    # if a site has too few neighbors, increase the cutoff radius
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)
    #print("min_nbrs:",min_nbrs)
    #print("use_canonize:",use_canonize)
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

    #############################
    This code comes from the following GitHub code.
    https://github.com/usnistgov/alignn/blob/main/alignn/graphs.py#L130
    #############################

    """
    # second pass: construct *undirected* graph
    # import pprint
    u, v, r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            d = atoms.lattice.cart_coords(
                dst_coord - atoms.frac_coords[src_id]
            )
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


def canonize_edge(
    src_id,
    dst_id,
    src_image,
    dst_image,
):
    """Compute canonical edge representation.

    Sort vertex ids
    shift periodic images so the first vertex is in (0,0,0) image

        
    #############################
    This code comes from the following GitHub code.
    https://github.com/usnistgov/alignn/blob/main/alignn/graphs.py#L28
    #############################

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


def build_undirected_edge_data_tensor(edges: defaultdict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            displace.append(-1*np.array(dst_image))

            # Adding the reverse edge for undirected graph
            src.append(dst_id)
            dst.append(src_id)
            displace.append(np.array(dst_image))

    # Convert lists to torch tensors
    src_ids = torch.tensor(src, dtype=torch.long)
    dst_ids = torch.tensor(dst, dtype=torch.long)
    displace = torch.tensor(displace, dtype=torch.float)

    return src_ids, dst_ids, displace







def compute_edge_vectors(dir_coords: torch.Tensor, 
                         lattice_vectors: torch.Tensor, 
                         src_ids: torch.Tensor, 
                         dst_ids: torch.Tensor, 
                         displace: torch.Tensor) -> torch.Tensor:
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