from typing import Tuple, Union

import dgl
import numpy as np
import torch
import torch.nn as nn

from alignn.models.alignn import (
    ALIGNNConfig,
    ALIGNNConv,
    EdgeGatedGraphConv,
    MLPLayer,
    RBFExpansion,
    AvgPooling
)


class ALIGNN4inverse(nn.Module):
    """
    Modified ALIGNN model for inverse materials design.

    This class implements a variant of the ALIGNN model that supports inverse design tasks.
    Instead of using graph node features directly from the graph, features like atomic distribution,
    bond lengths, and angles are provided explicitly as arguments to the forward function.

    Args:
        config (ALIGNNConfig): Configuration object containing model hyperparameters.

    Forward Args:
        g (Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]):
            A tuple of DGLGraph objects (crystal graph, line graph).
        atom_features (torch.Tensor):
            Tensor of shape (N, F_atom) representing atom features.
        bondlength (torch.Tensor):
            Tensor of shape (E,) representing bond lengths.
        angle_features (torch.Tensor):
            Tensor of shape (T, F_angle) representing angle-based features.

    Returns:
        torch.Tensor: Output predictions for each crystal in the batch.
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
