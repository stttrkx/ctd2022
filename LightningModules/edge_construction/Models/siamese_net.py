#!/usr/bin/env python
# coding: utf-8

"""Siamese network strategy using MPL, CNN and GAT as Subnetworks."""

# TODO: Implement Siamese Network strategy for Metric Learning.

import torch.nn.functional as F
from ..metric_base import MetricBase
from ..utils.embedding_utils import make_mlp


# Siamese Network with MLP
class SiameseMLP(MetricBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """Initialise the LightningModule to scan over different embedding training regimes"""

        # Select Input Regime
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]

        # Construct the MLP
        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x_out = self.network(x)
        if "norm" in self.hparams["regime"]:
            return F.normalize(x_out)
        else:
            return x_out


# Siamese Network with CNN
class SiameseCNN(MetricBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """Initialise the LightningModule to scan over different embedding training regimes"""

        # Select Input Regime
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]

        # Construct the MLP
        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x_out = self.network(x)
        if "norm" in self.hparams["regime"]:
            return F.normalize(x_out)
        else:
            return x_out


# Siamese Network with GAT
class SiameseGAT(MetricBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """Initialise the LightningModule to scan over different embedding training regimes"""

        # Select Input Regime
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]

        # Construct the MLP
        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x_out = self.network(x)
        if "norm" in self.hparams["regime"]:
            return F.normalize(x_out)
        else:
            return x_out
