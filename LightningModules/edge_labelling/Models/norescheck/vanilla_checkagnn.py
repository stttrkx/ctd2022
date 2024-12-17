#!/usr/bin/env python
# coding: utf-8
import torch
from torch_scatter import scatter_add
from torch.utils.checkpoint import checkpoint
from ..gnn_base import GNNBase
from ..utils.gnn_utils import make_mlp


# Attention GNN (AGNN) by P. Veličković [arXiv:1710.10903]


class VanillaCheckAGNN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        """
        The model `VanillaAGNN` is the attention model without a residual aka `skip`
        connection. It is the new implimentation of `GNNSegmentClassifier` model that
        was developed by Steven S. Farrell and presented in the CTD 2018 conference.
        """

        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )

        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )

        # Setup input network
        self.input_network = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # Setup edge network
        self.edge_network = make_mlp(
            (hparams["hidden"]) * 2,
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # Setup node network
        self.node_network = make_mlp(
            (hparams["hidden"]) * 2,
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

    def forward(self, x, edge_index):

        # senders, receivers
        start, end = edge_index

        # Apply input network
        x = self.input_network(x)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            # Apply edge network
            edge_inputs = torch.cat([x[start], x[end]], dim=1)
            e = checkpoint(self.edge_network, edge_inputs)
            e = torch.sigmoid(e)

            # Message-passing (aggregation) for unidirectional edges.
            # Old aggregation fixed for GNNBase when directed=True.
            # messages = scatter_add (
            #    e * x[start], end, dim=0, dim_size=x.shape[0]
            # ) + scatter_add (
            #    e * x[end], start, dim=0, dim_size=x.shape[0]
            # )

            # Message-passing (aggregation) for bidirectional edges.
            # New aggregation fixed for new GNNBase when directed=False.
            messages = scatter_add(
                # e[:, None] * x[start], end, dim=0, dim_size=x.shape[0]
                e * x[start],
                end,
                dim=0,
                dim_size=x.shape[0],
            )

            # Apply node network
            node_inputs = torch.cat([messages, x], dim=1)
            x = checkpoint(self.node_network, node_inputs)

        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return checkpoint(self.edge_network, edge_inputs)
