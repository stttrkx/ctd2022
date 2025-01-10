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
# Once complete, it will replace EmbeddingBase as the base class everywhere.

import os
import torch
import logging
from typing import Optional, Tuple, List, Dict, Any
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from .utils.embedding_utils import split_datasets, build_edges, graph_intersection

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="training.log",
    filemode="w",  # 'w' to overwrite, 'a' to append
)
logger = logging.getLogger(__name__)


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

        # Save hyperparameters
        self.save_hyperparameters(hparams)

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

    # Append Hard Negative Mining (hnm) with KNN graph
    def append_hnm_pairs(self, e_spatial, query, query_indices, spatial):
        """Append Hard Negative Mining (hnm) with KNN graph"""

        if "low_purity" in self.hparams["regime"]:
            knn_edges = build_edges(
                query, spatial, query_indices, self.hparams["r_train"], 500
            )
            knn_edges = knn_edges[
                :,
                torch.randperm(knn_edges.shape[1])[
                    : int(self.hparams["r_train"] * len(query))
                ],
            ]

        else:
            knn_edges = build_edges(
                query,
                spatial,
                query_indices,
                self.hparams["r_train"],
                self.hparams["knn"],
            )

        e_spatial = torch.cat(
            [
                e_spatial,
                knn_edges,
            ],
            dim=-1,
        )

        return e_spatial

    # Append Random Edges Pairs (rp) for Stability
    def append_random_pairs(self, e_spatial, query_indices, spatial):
        """Append Random Edges Pairs (rp) for Stability."""

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
    def get_truth(self, batch, e_spatial, e_bidir):
        """Calculate truth from intersection between Prediction graph and Truth graph"""
        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        return e_spatial, y_cluster

    # Append all Positive Examples and Their Truth and Weighting
    def get_true_pairs(self, e_spatial, y_cluster, e_bidir, new_weights):
        """
        Incorporate ground truth edges into the current edge set and update corresponding labels and weights.

        This function ensures that all known true positive edges (ground truth) are included in the edge set
        used for training. It updates the edge list, labels, and weights to reflect the addition of these
        known positive examples.

        Args:
            e_spatial (torch.Tensor): Current edge list, which may include both predicted and some true edges.
            y_cluster (torch.Tensor): Binary labels for the current edges (1 for positive, 0 for negative).
            e_bidir (torch.Tensor): Bidirectional ground truth edges.
            new_weights (torch.Tensor): Current weights assigned to each edge.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Updated edge list (e_spatial) including all ground truth edges
                - Updated binary labels (y_cluster) for all edges
                - Updated weights (new_weights) for all edges
        """

        # Append ground truth edges (e_bidir) to the current edge list (e_spatial)
        # This ensures all known true positive pairs are included in the training process
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            dim=-1,
        )

        # Update labels to reflect the addition of ground truth edges
        # All newly added edges from e_bidir are positive examples, so we append a tensor of ones
        y_cluster = torch.cat(
            [y_cluster.int(), torch.ones(e_bidir.shape[1], device=self.device)]
        )

        # Assign weights to the newly added ground truth edges
        # We use a predefined weight (self.hparams["weight"]) for these known positive examples
        # This typically gives more importance to the ground truth edges in the loss calculation
        new_weights = torch.cat(
            [
                new_weights,
                torch.ones(e_bidir.shape[1], device=self.device)
                * self.hparams["weight"],
            ]
        )

        return e_spatial, y_cluster, new_weights

    # Weighted Hinge Loss
    def get_hinge_distance(self, spatial, e_spatial, y_cluster):
        """Compute squared Euclidean distances between pairs of points and hinge labels.
        Args:
            spatial (torch.Tensor): Node embeddings of shape (num_nodes, embedding_dim).
            e_spatial (torch.Tensor): Edge indices of shape (2, num_edges).
            y_cluster (torch.Tensor): Edge labels (1 for similar, 0 for dissimilar) of shape (num_edges,).
        Returns:
            hinge (torch.Tensor): Hinge labels (1 for similar, -1 for dissimilar) of shape (num_edges,).
            d (torch.Tensor): Squared distances between reference and neighbour nodes of shape (num_edges,).
        """

        # Convert actual labels (0, 1) to hinge labels (-1, +1)
        hinge = 2 * y_cluster.float().to(self.device) - 1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])

        try:  # This can be resource intensive, so we chunk it if it fails
            d_sq = torch.sum((reference - neighbors) ** 2, dim=-1)
        except RuntimeError:
            d_sq = [
                torch.sum((ref - nei) ** 2, dim=-1)
                for ref, nei in zip(reference.chunk(10), neighbors.chunk(10))
            ]
            d_sq = torch.cat(d_sq)

        return hinge, d_sq

    def get_weighted_hinge_loss(self, hinge, d, weight, verbose=False):
        """Compute the weighted hinge loss for the embedding model. It uses "mean" reduction in
        hinge loss calculation, which is computationally more efficient. However, it only applies
        weight to similar edges. No way to adjust weights for dissimilar edges (disadvantage).
        Args:
            hinge (torch.Tensor): Hinge labels (1 for similar, -1 for dissimilar) of shape (num_edges,).
            d (torch.Tensor): Squared distances between reference and neighbour nodes of shape (num_edges,).
            weight (float): Scalar weight hyperparameter affecting similar edges only.
        Returns:
            torch.Tensor: The total weighted hinge loss.
        """

        # Hangle hinge margin (default: 0.1)
        hinge_margin = self.hparams.get("margin", 0.1) ** 2

        # Negative loss: Push dissimilar pairs apart (hinge == -1)
        negative_mask = hinge == -1
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[negative_mask],
            hinge[negative_mask],
            margin=hinge_margin,
            reduction="mean",
        )

        # Positive loss: Pull similar pairs closer (hinge == 1)
        positive_mask = hinge == 1
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[positive_mask],
            hinge[positive_mask],
            margin=hinge_margin,
            reduction="mean",
        )
        positive_loss = weight * positive_loss

        # Weighted hinge loss
        weighted_loss = negative_loss + positive_loss

        # Print loss values for debugging
        if verbose:
            logger.info(f"Positive Loss: {positive_loss.item()}")
            logger.info(f"Negative Loss: {negative_loss.item()}")
            logger.info(f"Weighted Total Loss: {weighted_loss.item()}")

        return weighted_loss

    # --------------------------- Training Functions ---------------------------
    # Training Step
    def training_step(self, batch, batch_idx):
        """Training step for the embedding model by using weighted hinge loss. We use
        two approaches for calculating the loss: one uses a single scalar weight that
        only affects positive pairs ("mean" reduction before weighting), while the other
        uses per-edge weights through a weight tensor ("none" reduction and applies
        weights before mean). Both utilize a scalar "weight" from the config file. We've
        to adopt one of the two approaches, but both are provided here for reference.
        can use only one of the two approaches by commenting out the other one."""

        # Create an empty edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Get input data
        input_data = self.get_input_data(batch)

        # Get node embeddings
        with torch.no_grad():
            spatial = self(input_data)

        # Get query points
        query_indices, query = self.get_query_points(batch, spatial)

        # Append hard negative mining (hnm) with KNN graph
        e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial)

        # Append random edges pairs (rp) for stability
        e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial)

        # Instantiate bidirectional truth, since KNN prediction is bidirectional
        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], dim=-1
        )

        # Get labelled graph from intersection between prediction and truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        # Create weight tensor by giving self.hparams["weight"] to similar
        # pairs and 0 to the dissimilar pairs, useful for per-edge weighting
        weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, e_bidir, weights
        )

        # Select unique edges
        included_hits = e_spatial.unique()
        spatial[included_hits] = self(input_data[included_hits])

        # Get squared Euclidean distance between pairs and hinge labels
        hinge, d = self.get_hinge_distance(spatial, e_spatial, y_cluster)

        # Give dissimlar pairs a weight of 1 (y_cluster==0 or hinge==-1).
        # Note that there may still be TRUE examples that are weightless
        new_weights[hinge == -1] = 1  # no contribution from dissimlar pairs

        # Weighted hinge loss where weight tensor affects all edges.
        # d = d * new_weights # early weighting to use "mean" reduction
        # loss = torch.nn.functional.hinge_embedding_loss(
        #    d, hinge, margin=self.hparams["margin"], reduction="mean"
        # )

        # Or, Weighted hinge loss where a scalar weight affects similar edges.
        loss = self.get_weighted_hinge_loss(hinge, d, self.hparams["weight"])

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
        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], dim=-1
        )

        # Build whole KNN graph
        e_spatial = build_edges(
            spatial, spatial, indices=None, r_max=knn_rad, k_max=knn_num
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        # Calculate hinge distance
        hinge, d = self.get_hinge_distance(
            spatial, e_spatial.to(self.device), y_cluster
        )

        # Calculate hinge loss
        loss = F.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"] ** 2, reduction="mean"
        )

        # Calculate efficiency and purity
        cluster_true = e_bidir.shape[1]
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
            logger.info("Efficiency: {}".format(eff))
            logger.info("Purity: {}".format(pur))
            logger.info(batch.event_file)

        return {
            "loss": loss,
            "distances": d,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
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
