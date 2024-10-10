
# This file includes code from the National Institute of Standards and Technology (NIST).
# See the NIST license information in the LICENSE file.

import json
import os
import zipfile
from typing import Tuple, Union

import dgl
import numpy as np
import requests
import torch
from dgl.nn import AvgPooling
from torch import nn
from tqdm import tqdm

from alignn.models.alignn import ALIGNNConfig, ALIGNNConv, EdgeGatedGraphConv, MLPLayer
from alignn.models.utils import RBFExpansion


class ALIGNN4inverse(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.

    <Note>
    This code is based on the following GitHub code. However, it has been modified to be used for the inverse problem.
    https://github.com/usnistgov/alignn/blob/main/alignn/models/alignn.py#L191

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
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
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
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph], 
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
            '''Commented out and changed for inverse problem
            # angle features (fixed)    
            z = self.angle_embedding(lg.edata.pop("h"))
            '''
            z = self.angle_embedding(angle_features)

        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            # print('g',g)
            # print('features1',features.shape)
            features = self.extra_feature_embedding(features)

        ''' Commented out and changed for inverse problem
        g = g.local_var()
        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        # print('x1',x.shape)
        x = self.atom_embedding(x)
        # print('x2',x.shape)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)'''
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
    

def figshare_model_list():
    return {
        "mp_e_form_alignnn": [
            "https://figshare.com/ndownloader/files/31458811",
            1,
        ],
        "mp_gappbe_alignnn": [
            "https://figshare.com/ndownloader/files/31458814",
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
    path = str(os.path.join(os.path.dirname('__file__'), zfile))
    #path = str(os.path.join(os.path.dirname(__file__), zfile)) # 非対話モードであれば、こちらを使う
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
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

    #new_file, filename = tempfile.mkstemp()
    filename = os.path.join("./models/tmp.pt")
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)

    return model


def Load_Pretrained_ALIGNN(prediction_loss_setting_dict:dict)->dict:
    '''
    Load the pretrained ALIGNN model for the inverse problem.
    Args:
        prediction_loss_setting_dict: The dictionary containing the settings for the prediction loss.
    Returns:
        prediction_loss_setting_dict: The dictionary containing the settings for the prediction loss with the pretrained ALIGNN model.
    '''

    for loss_key in prediction_loss_setting_dict.keys():
        prediction_loss_setting_dict[loss_key]['prediction_model'] = get_figshare_model_for_inverse_problem(model_name=prediction_loss_setting_dict[loss_key]['alignn_model_name'])
    return prediction_loss_setting_dict