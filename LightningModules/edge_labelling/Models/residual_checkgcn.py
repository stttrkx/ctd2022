#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint

from ..gnn_base import GNNBase
from ..utils.gnn_utils import make_mlp


# Checkpointed Residual AGNN
class CheckResGCN(GNNBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        The model `CheckResGCN` is the graph convolutional network proposed by 
        Thomas Kipf in his paper [arXiv:1609.02907]. In includes residual/skip
        connection. It is tested for reconstruction by Exa.TrkX collaboration.
        """

        concatenation_factor = (
            3
            if (self.hparams["aggregation"] in ["sum_max", "mean_max", "mean_sum"])
            else 2
        )
        hparams["output_activation"] = (
            None if "output_activation" not in hparams else hparams["output_activation"]
        )
        hparams["batchnorm"] = (
            False if "batchnorm" not in hparams else hparams["batchnorm"]
        )

        # Setup input network
        self.node_encoder = make_mlp(
            hparams["spatial_channels"] + hparams["cell_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The edge network computes new edge features from connected nodes
        self.edge_network = make_mlp(
            2 * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

        # The node network computes new node features
        self.node_network = make_mlp(
            concatenation_factor * (hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_node_layer"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )

    def forward(self, x, edge_index):

        # Senders and receivers
        start, end = edge_index

        # Apply input network
        x = self.node_encoder(x)
        x = F.softmax(x, dim=-1)

        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            # Residual connection
            x_initial = x

            # Message-passing (aggregation) for unidirectional edges.
            # Old aggregation fixed for GNNBase when directed=True.
            # edge_messages = scatter_add(
            #    x[start], end, dim=0, dim_size=x.shape[0]
            # ) + scatter_add(x[end], start, dim=0, dim_size=x.shape[0])

            # Message-passing (aggregation) for bidirectional edges.
            # New aggregation fixed for new GNNBase when directed=False.
            # edge_messages = scatter_add(  # sum
            #    x[start], end, dim=0, dim_size=x.shape[0]
            # )

            # aggregation: sum, mean, max, sum_max, mean_sum, mean_max
            edge_messages = None
            if self.hparams["aggregation"] == "sum":
                edge_messages = scatter_add(x[start], end, dim=0, dim_size=x.shape[0])
            elif self.hparams["aggregation"] == "max":
                edge_messages = scatter_max(x[start], end, dim=0, dim_size=x.shape[0])[
                    0
                ]
            elif self.hparams["aggregation"] == "sum_max":
                edge_messages = torch.cat(
                    [
                        scatter_max(x[start], end, dim=0, dim_size=x.shape[0])[0],
                        scatter_add(x[start], end, dim=0, dim_size=x.shape[0]),
                    ],
                    dim=-1,
                )
            elif self.hparams["aggregation"] == "mean":
                edge_messages = scatter_mean(x[start], end, dim=0, dim_size=x.shape[0])
            elif self.hparams["aggregation"] == "mean_sum":
                edge_messages = torch.cat(
                    [
                        scatter_mean(x[start], end, dim=0, dim_size=x.shape[0]),
                        scatter_add(x[start], end, dim=0, dim_size=x.shape[0]),
                    ],
                    dim=-1,
                )
            elif self.hparams["aggregation"] == "mean_max":
                edge_messages = torch.cat(
                    [
                        scatter_max(x[start], end, dim=0, dim_size=x.shape[0])[0],
                        scatter_mean(x[start], end, dim=0, dim_size=x.shape[0]),
                    ],
                    dim=-1,
                )

            # Apply node network
            node_inputs = torch.cat([x, edge_messages], dim=-1)
            node_inputs = F.softmax(node_inputs, dim=-1)
            x = checkpoint(self.node_network, node_inputs)

            # Residual connection
            x = x + x_initial

        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return checkpoint(self.edge_network, edge_inputs)
