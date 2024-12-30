#!/usr/bin/env python
# coding: utf-8

"""The base classes for the embedding process:
The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is 
a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one 
of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss 
is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style 
losses. Also available are triplet loss approaches.

Example: See Quickstart for a concrete example of this process.
Todo: Refactor the training & validation steps, since the use of different regimes (rp, hnm, etc.) looks very messy.
"""

# TODO: MetricBase is exactly the same as EmbeddingBase. I intend to develop it
# in PyTorch & PyTorch Lightning 2.x, hence I keep it separate from EmbeddingBase.

import os
import torch
import logging
from typing import Optional, Tuple, List, Dict, Any
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from .utils.embedding_utils import graph_intersection, split_datasets, build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class MetricBase(LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        """The LightningModule to scan over different embedding training regimes"""

        # Handle Workers
        self.n_workers = (
            self.hparams["n_workers"]
            if "n_workers" in self.hparams
            else len(os.sched_getaffinity(0))
        )

        # Handle Datasets
        self.trainset, self.valset, self.testset = None, None, None

        # Handle Model
        in_channels = hparams["spatial_channels"]
        self.network = make_mlp(
            in_channels,
            [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
            hidden_activation=hparams["activation"],
            output_activation=None,
            layer_norm=True,
        )

        # Save hyperparameters
        self.save_hyperparameters(hparams)

    def forward(self, x):
        x_out = self.network(x)
        return F.normalize(x_out)

    # Load Data
    def setup(self, stage: Optional[str] = "fit") -> None:
        self.trainset, self.valset, self.testset = split_datasets(**self.hparams)

    # DataLoader Hooks
    def train_dataloader(self) -> Optional[DataLoader]:
        """Training DataLoader Hook"""
        if self.trainset is not None:
            return DataLoader(
                self.trainset,
                batch_size=1,
                num_workers=self.n_workers,
                # , pin_memory=True, persistent_workers=True
            )
        else:
            return None

    def val_dataloader(self) -> Optional[DataLoader]:
        """Validation DataLoader Hook"""
        if self.valset is not None:
            return DataLoader(
                self.valset,
                batch_size=1,
                num_workers=self.n_workers,
                # , pin_memory=True, persistent_workers=True
            )
        else:
            return None

    def test_dataloader(self) -> Optional[DataLoader]:
        """Test DataLoader Hook"""
        if self.testset is not None:
            return DataLoader(
                self.testset,
                batch_size=1,
                num_workers=self.n_workers,
                # , pin_memory=True, persistent_workers=True
            )
        else:
            return None

    def predict_dataloader(self) -> Optional[DataLoader]:
        """Predict DataLoader Hook"""
        return [
            self.train_dataloader(),
            self.val_dataloader(),
            self.test_dataloader(),
        ]

    # Optimizer and Scheduler
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    # def on_train_start(self):
    #    self.trainer.strategy.optimizers = [
    #        self.trainer.lr_scheduler_configs[0].scheduler.optimizer
    #    ]

    # ---------------------------- Helper Functions ----------------------------

    # Get input data
    def get_input_data(self, batch: Any) -> torch.Tensor:
        """Get input data, handling whether Cell Information (ci) is included"""

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], dim=-1
            )
            input_data[input_data != input_data] = 0  # replace NaNs with 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0  # replace NaNs with 0

        return input_data

    # Get query points
    def get_query_points(
        self, batch: Any, spatial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get query points based on the regime"""

        if "query_all_points" in self.hparams["regime"]:
            query_indices = torch.arange(len(spatial)).to(spatial.device)
        elif "query_noise_points" in self.hparams["regime"]:
            query_indices = torch.cat(
                [torch.where(batch.pid == 0)[0], batch.signal_true_edges.unique()]
            )
        else:
            query_indices = batch.signal_true_edges.unique()

        query_indices = query_indices[torch.randperm(len(query_indices))][
            : self.hparams["points_per_batch"]
        ]
        query = spatial[query_indices]

        return query_indices, query

    # Append Hard Negative Mining (hnm) Pairs with Prediction Edge List
    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial):
        """Append Hard Negative Mining (hnm) with KNN graph"""

        knn_edges = build_edges(
            query,
            spatial,
            query_indices,
            self.hparams["r_train"],
            self.hparams["knn"],
        )

        # append kNN edges to edge list
        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            dim=-1,
        )

        # return final edge list
        return e_spatial

    # Append Random Pairs to Prediction Edge List
    def append_random_pairs(self, e_spatial, query_indices, spatial):
        """Append random edges pairs (rp) for stability"""

        n_random = int(self.hparams["randomisation"] * len(query_indices))
        indices_src = torch.randint(
            0, len(query_indices), (n_random,), device=self.device
        )
        indices_dest = torch.randint(0, len(spatial), (n_random,), device=self.device)
        random_pairs = torch.stack([query_indices[indices_src], indices_dest])

        e_spatial = torch.cat(
            [e_spatial, random_pairs],
            dim=-1,
        )
        return e_spatial

    # Calculate truth from intersection between Prediction graph and Truth graph
    def get_truth(self, batch, e_spatial, true_edges_bidir):
        """Calculate truth from intersection between Prediction graph and Truth graph"""
        e_spatial, y_cluster = graph_intersection(e_spatial, true_edges_bidir)

        return e_spatial, y_cluster

    # Append all positive examples and their truth and weighting
    def get_true_pairs(self, e_spatial, y_cluster, new_weights, true_edges_bidir):
        """
        Incorporate ground truth edges into the current edge set and update corresponding labels and weights.

        This function ensures that all known true positive edges (ground truth) are included in the edge set
        used for training. It updates the edge list, labels, and weights to reflect the addition of these
        known positive examples.

        Args:
            e_spatial (torch.Tensor): Current edge list, which may include both predicted and some true edges.
            y_cluster (torch.Tensor): Binary labels for the current edges (1 for positive, 0 for negative).
            new_weights (torch.Tensor): Current weights assigned to each edge.
            true_edges_bidir (torch.Tensor): Bidirectional ground truth edges.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Updated edge list (e_spatial) including all ground truth edges
                - Updated binary labels (y_cluster) for all edges
                - Updated weights (new_weights) for all edges
        """
        # Append ground truth edges (true_edges_bidir) to the current edge list (e_spatial)
        # This ensures all known true positive pairs are included in the training process
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                true_edges_bidir,
            ],
            dim=-1,
        )

        # Update labels to reflect the addition of ground truth edges
        # All newly added edges from true_edges_bidir are positive examples, so we append a tensor of ones
        y_cluster = torch.cat(
            [y_cluster.int(), torch.ones(true_edges_bidir.shape[1], device=self.device)]
        )

        # Assign weights to the newly added ground truth edges
        # We use a predefined weight (self.hparams["weight"]) for these known positive examples
        # This typically gives more importance to the ground truth edges in the loss calculation
        new_weights = torch.cat(
            [
                new_weights,
                torch.ones(true_edges_bidir.shape[1], device=self.device)
                * self.hparams["weight"],
            ]
        )

        return e_spatial, y_cluster, new_weights

    # Custom Loss Functions
    def loss_function(
        self, batch, spatial, e_spatial=None, y_cluster=None, weights=None
    ):
        """Steering function to compute loss (hinge, triplet or cosine).

        Args:
            spatial (torch.Tensor): Node embeddings of shape (num_nodes, embedding_dim).
            e_spatial (torch.Tensor): Edge indices of shape (2, num_edges).
            y_cluster (torch.Tensor): Edge labels (1 for similar, 0 for dissimilar) of shape (num_edges,).
            weights (torch.Tensor, optional): Weights for each edge of shape (num_edges,).
                Positive weights maintain relationship, negative weights flip it, zero weights exclude pairs.
        Returns:
            torch.Tensor: The total weighted loss (hinge, triplet or cosine).
        """

        # Compute squared Euclidean distances between pairs of points
        d = self.pairwise_distances(spatial, e_spatial)

        # Return weighted hinge loss
        return self.get_weighted_hinge_loss(y_cluster, d, weights)

    def pairwise_distances(self, spatial, e_spatial):
        """
        Compute squared Euclidean distances between pairs of points.

        Args:
            spatial (torch.Tensor): Node embeddings of shape (num_nodes, embedding_dim).
            e_spatial (torch.Tensor): Edge indices of shape (2, num_edges).
        Returns:
            d (torch.Tensor): Squared distances between reference and neighbor points.
        """
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        try:  # This can be resource intensive, so we chunk it if it fails
            d = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            d = [
                torch.sum((ref - nei) ** 2, dim=-1)
                for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))
            ]
            d = torch.cat(d)

        return d

    def get_weighted_hinge_loss(self, y_cluster, d, weights):
        """Compute the weighted hinge loss for an embedding model. It converts
        actual edge labels (0, 1) to hinge labels (-1, 1) for computing loss.

        Args:
            y_cluster (torch.Tensor): Edge labels (1 for similar, 0 for dissimilar) of shape (num_edges,).
            d (torch.Tensor): Squared distances between reference and neighbor points.
            weights (torch.Tensor): Weights for each edge of shape (num_edges,).
                Positive weights maintain relationship, negative weights flip it, zero weights exclude pairs.
        Returns:
            torch.Tensor: The total weighted hinge loss.
        """
        # FIXME: Simple separate loss calculation code from the training_step()
        # without changing the training logic, understand how hinge margin, and
        # edge weights play a role in computing the weighted hinge loss.

        # Convert actual labels (0, 1) to hinge labels (-1, +1)
        hinge = 2 * y_cluster.float().to(self.device) - 1

        # Handle edge weights (default: 2.0)
        weights = torch.ones_like(d) if weights is None else weights

        # Hangle hinge margin (default: 0.1)
        margin_squared = self.hparams["margin"] ** 2

        # Negative loss: Push dissimilar pairs apart (hinge == -1)
        negative_mask = hinge == -1
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[negative_mask],
            hinge[negative_mask],
            margin=margin_squared,
            reduction="none",
        )
        negative_loss = (negative_loss * weights[negative_mask]).mean()

        # Positive loss: Pull similar pairs closer (hinge == 1)
        positive_mask = hinge == 1
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[positive_mask],
            hinge[positive_mask],
            margin=margin_squared,
            reduction="none",
        )
        positive_loss = (positive_loss * weights[positive_mask]).mean()

        # Return total weighted hinge loss
        return negative_loss + positive_loss

    def weighted_hinge_loss(self, y_cluster, d, weights):
        """Compute the weighted hinge loss for an embedding model. It converts
        actual edge labels (0, 1) to hinge labels (-1, 1) for computing loss.

        Args:
            y_cluster (torch.Tensor): Edge labels (1 for similar, 0 for dissimilar) of shape (num_edges,).
            d (torch.Tensor): Squared distances between reference and neighbor points.
            weights (torch.Tensor): Weights for each edge of shape (num_edges,).
                Positive weights maintain relationship, negative weights flip it, zero weights exclude pairs.
        Returns:
            torch.Tensor: The total weighted hinge loss.
        """
        # TODO: Make it competible with weighted_hinge_loss_new(), problem is the
        # weighting scheme used, in our case self.hparams["weight"] is set to 2.

        # Convert actual labels (0, 1) to hinge labels (-1, +1)
        hinge = 2 * y_cluster.float().to(self.device) - 1

        # Handle edge weights (default: 2.0)
        weights = torch.ones_like(d) if weights is None else weights

        # Hangle hinge margin (default: 0.1)
        margin_squared = self.hparams["margin"] ** 2

        # Negative mask to handle weight signs and zeros
        negative_mask = ((hinge == -1) & (weights > 0)) | ((hinge == 1) & (weights < 0))
        negative_mask = negative_mask & (weights != 0)

        # Negative loss: Push dissimilar pairs apart
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[negative_mask],
            torch.ones_like(d[negative_mask]) * -1,  # Always use -1
            margin=margin_squared,
            reduction="none",
        )
        # Use absolute weights
        negative_loss = (negative_loss * weights[negative_mask].abs()).mean()

        # Positive mask to handle weight signs and zeros
        positive_mask = ((hinge == 1) & (weights > 0)) | ((hinge == -1) & (weights < 0))
        positive_mask = positive_mask & (weights != 0)

        # Positive loss: Pull similar pairs closer
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[positive_mask],
            torch.ones_like(d[positive_mask]) * 1,  # Always use 1
            margin=margin_squared,
            reduction="none",
        )
        # Use absolute weights
        positive_loss = (positive_loss * weights[positive_mask].abs()).mean()

        # Return total weighted hinge loss
        return negative_loss + positive_loss

    def weighted_hinge_loss_new(self, y_cluster, d, weights):
        """Compute the weighted hinge loss for an embedding model. It uses actual
        edge labels (0, 1) and converts them into hinge labels (-1, 1) on the fly.

        Args:
            y_cluster (torch.Tensor): Edge labels (1 for similar, 0 for dissimilar) of shape (num_edges,).
            d (torch.Tensor): Squared distances between reference and neighbor points.
            weights (torch.Tensor): Weights for each edge of shape (num_edges,).
                Positive weights maintain relationship, negative weights flip it, zero weights exclude pairs.
        Returns:
            torch.Tensor: The total weighted hinge loss.
        """
        # FIXME: What is weighting scheme here?

        # Negative mask to push dissimilar pairs apart
        negative_mask = ((y_cluster == 0) & (weights != 0)) | (weights < 0)

        # Handle negative loss, but don't reduce vector
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[negative_mask],
            torch.ones_like(d[negative_mask]) * -1,
            margin=self.hparams["margin"] ** 2,
            reduction="none",
        )
        # Now reduce the vector with non-zero weights
        negative_loss = (negative_loss * weights[negative_mask].abs()).mean()

        # Positive mask to pull similar pairs closer
        positive_mask = (y_cluster == 1) & (weights > 0)

        # Handle positive loss, but don't reduce vector
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[positive_mask],
            torch.ones_like(d[positive_mask]),
            margin=self.hparams["margin"] ** 2,
            reduction="none",
        )
        # Now reduce the vector with non-zero weights
        positive_loss = (positive_loss * weights[positive_mask].abs()).mean()

        # Return total weighted hinge loss
        return negative_loss + positive_loss

    # --------------------------- Training Functions ---------------------------

    # Training Step
    def training_step(self, batch, batch_idx):
        """Training step"""
        # TODO: Training Steps
        # 1 - Get input data
        # 2 - Get node embeddings (spatial/embedding)
        # 3 - Build edges (e_spatial)
        # 4 - Get edges and labels (e_spatial, y_cluster) <-- get_truth()
        # 5 - Handle edge weighting (new e_spatial, y_cluster, etc) <-- get_true_pairs()
        # 6 - Compute, log and return weighted hinge loss

        # Empty edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Get input data
        input_data = self.get_input_data(batch)

        # Get node embeddings
        with torch.no_grad():
            spatial = self(input_data)

        # Get Query Points
        query_indices, query = self.get_query_points(batch, spatial)

        # Append Hard Negative Mining (hnm) with KNN graph
        e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial)

        # Append Random Edges Pairs (rp) for Stability
        e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial)

        # FIXME: Simplify logic by using more
        # Get Signal True Edges (bidirectional since KNN prediction will be bidirectional)
        true_edges_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], dim=-1
        )

        # Get Labelled Edge List (e_spatial, y_cluster) using Graph Intersection
        # Note: e_spatial is edge list we constructed using hnm & rp, we need
        # ground truth of our graph which is true_edges_bidir (find a better name) here.
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, true_edges_bidir)

        # Calculate Weights
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, true_edges_bidir
        )

        included_hits = e_spatial.unique()

        spatial[included_hits] = self(input_data[included_hits])

        # Calculate Hinge Loss

        # Get distance between the reference and neighbor points
        d = self.pairwise_distances(spatial, e_spatial)

        # Convert actual labels (0, 1) to hinge labels (-1, +1)
        hinge = 2 * y_cluster.float().to(self.device) - 1

        # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        new_weights[hinge == -1] = 1

        # Calculate negative loss
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"] ** 2,
            reduction="mean",
        )

        # Calculate positive loss
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"] ** 2,
            reduction="mean",
        )

        # Calculate total loss
        loss = negative_loss + self.hparams["weight"] * positive_loss

        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=1)

        return loss

    # Shared evaluation function
    def shared_evaluation(
        self,
        batch,
        batch_idx,
        knn_rad,
        knn_num,
        log=False,
        verbose=False,
    ):
        """Shared Evaluation Function for Validation and Test Steps"""

        # Get input data
        input_data = self.get_input_data(batch)

        # Get spatial embeddings (forward pass)
        spatial = self(input_data)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        true_edges_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], dim=-1
        )

        # Build whole KNN graph
        e_spatial = build_edges(
            spatial, spatial, indices=None, r_max=knn_rad, k_max=knn_num
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, true_edges_bidir)

        # Calculate hinge distance
        hinge, d = self.get_hinge_distance(
            spatial, e_spatial.to(self.device), y_cluster
        )

        # Calculate hinge loss
        loss = F.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"] ** 2, reduction="mean"
        )

        # Calculate efficiency and purity
        cluster_true = true_edges_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        # Calculate efficiency and purity
        eff = cluster_true_positive / cluster_true
        pur = cluster_true_positive / cluster_positive

        # Log metrics
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "current_lr": current_lr},
                on_epoch=True,
                on_step=False,
                batch_size=1,
            )

        # Print metrics
        if verbose:
            logging.info("Efficiency: {}".format(eff))
            logging.info("Purity: {}".format(pur))
            logging.info(batch.event_file)

        return {
            "loss": loss,
            "distances": d,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": true_edges_bidir,
            "eff": eff,
            "pur": pur,
        }

    # Validation Step
    def validation_step(self, batch, batch_idx):
        """Run Validation Loop"""
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], 150, log=True
        )

        return outputs["loss"]

    # Test Step
    def test_step(self, batch, batch_idx):
        """Run Testing Loop"""
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 1000, log=False
        )

        return outputs

    # Predict Step
    def predict_step(self, batch, batch_idx):
        """Run Prediction Loop"""
        pass

    # Optimizer Step
    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Settings before Optimizer Step"""

        # warm up lr
        if self.hparams.get("warmup", 0) and (
            self.trainer.current_epoch < self.hparams["warmup"]
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # after reaching minimum learning rate, stop LR decay
        for pg in optimizer.param_groups:
            pg["lr"] = max(pg["lr"], self.hparams.get("min_lr", 0))
