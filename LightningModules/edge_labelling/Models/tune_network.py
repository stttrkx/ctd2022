#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..tune_base import TuneBase
from ..utils.dnn_utils import make_mlp


class EdgeClassifier_Tune(TuneBase):
    """A Dense Network for Edge Classification. Norm Options: Layer or Batch Norms."""

    def __init__(self, hparams):
        super().__init__(hparams)

        # Input Size: 2*(Node Features)
        self.input_dim = hparams["spatial_channels"] + hparams["cell_channels"]

        self.layer_dim = [
            hparams["l1_size"],
            hparams["l2_size"],
            hparams["l3_size"],
            hparams["l4_size"],
            # hparams["l5_size"],
            1,
        ]

        self.hidden_activation = hparams["hidden_activation"]
        self.layer_norm = self.hparams["layernorm"]
        self.batch_norm = self.hparams["batchnorm"]

        # Create a Dense Network
        self.dense = make_mlp(
            input_size=self.input_dim * 2,  # Features
            sizes=self.layer_dim,  # Nodes
            hidden_activation=self.hidden_activation,  # Relu
            output_activation=None,  # None
            layer_norm=self.layer_norm,  # LayerNorm
            batch_norm=self.batch_norm,  # BatchNorm
        )

    def forward(self, x, edge_index):

        # split edge_index
        start, end = edge_index

        # get edge features
        edge_inputs = torch.cat([x[start], x[end]], dim=1)

        return self.dense(edge_inputs)
