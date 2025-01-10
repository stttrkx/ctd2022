#!/usr/bin/env python
# coding: utf-8

import torch.nn.functional as F
from ..super_embedding_base import SuperEmbeddingBase
from ..utils.embedding_utils import make_mlp


class SuperLayerlessEmbedding(SuperEmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        # Construct the MLP architecture
        if "ci" in hparams["regime"]:
            in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
        else:
            in_channels = hparams["spatial_channels"]

        self.network1 = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.network2 = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        self.save_hyperparameters()

    def forward(self, x):

        x1_out = self.network1(x)
        x2_out = self.network2(x)

        return F.normalize(x1_out), F.normalize(x2_out)
