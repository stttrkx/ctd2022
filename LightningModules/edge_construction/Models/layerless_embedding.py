#!/usr/bin/env python
# coding: utf-8

import torch.nn.functional as F
from ..embedding_base import EmbeddingBase
from ..utils.embedding_utils import make_mlp


class LayerlessEmbedding(EmbeddingBase):
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
        return F.normalize(x_out)
