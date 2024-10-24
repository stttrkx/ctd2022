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
from pytorch_lightning import LightningModule
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from .utils.embedding_utils import graph_intersection, split_datasets, build_edges

device = "cuda" if torch.cuda.is_available() else "cpu"


class MetricBase(LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super().__init__()
        """The LightningModule to scan over different embedding training regimes"""

        # Save hyperparameters
        self.save_hyperparameters(hparams)

        # Set workers from hparams
        self.n_workers = (
            self.hparams["n_workers"]
            if "n_workers" in self.hparams
            else len(os.sched_getaffinity(0))
        )

        # Instance Variables
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage: Optional[str] = "fit") -> None:
        self.trainset, self.valset, self.testset = split_datasets(**self.hparams)

    # Training DataLoader
    def train_dataloader(self) -> Optional[DataLoader]:
        """Get the training dataloader"""
        if self.trainset is not None:
            return DataLoader(
                self.trainset,
                batch_size=1,
                num_workers=self.n_workers,
                # , pin_memory=True, persistent_workers=True
            )
        else:
            return None

    # Validation DataLoader
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get the validation dataloader"""
        if self.valset is not None:
            return DataLoader(
                self.valset,
                batch_size=1,
                num_workers=self.n_workers,
                # , pin_memory=True, persistent_workers=True
            )
        else:
            return None

    # Test DataLoader
    def test_dataloader(self) -> Optional[DataLoader]:
        """Get the test dataloader"""
        if self.testset is not None:
            return DataLoader(
                self.testset,
                batch_size=1,
                num_workers=self.n_workers,
                # , pin_memory=True, persistent_workers=True
            )
        else:
            return None

    # Optimizer and Scheduler
    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        """Configure the Optimizer and Scheduler"""
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

    # Before optimizer step
    def on_before_optimizer_step(self, optimizer, *args, **kwargs):
        """Settings before optimizer step"""

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

    # Helper Functions (1): Get input data
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

    # Helper Functions (2): Get query points
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

    # Helper Functions (3): Append Hard Negative Mining (hnm) Pairs with Prediction Graph
    def append_hnm_pairs(
        self,
        e_spatial: torch.Tensor,
        query: torch.Tensor,
        query_indices: torch.Tensor,
        spatial: torch.Tensor,
    ) -> torch.Tensor:
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

    # Helper Functions (4): Append Random Pairs to Prediction Graph
    def append_random_pairs(
        self,
        e_spatial: torch.Tensor,
        query_indices: torch.Tensor,
        spatial: torch.Tensor,
    ) -> torch.Tensor:
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

    # Helper Functions (5): Append all positive examples and their truth and weighting
    def get_true_pairs(
        self,
        e_spatial: torch.Tensor,
        y_cluster: torch.Tensor,
        new_weights: torch.Tensor,
        e_bidir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append all positive examples and their truth and weighting"""
        e_spatial = torch.cat(
            [
                e_spatial.to(self.device),
                e_bidir,
            ],
            dim=-1,
        )
        y_cluster = torch.cat([y_cluster.int(), torch.ones(e_bidir.shape[1])])
        new_weights = torch.cat(
            [
                new_weights,
                torch.ones(e_bidir.shape[1], device=self.device)
                * self.hparams["weight"],
            ]
        )
        return e_spatial, y_cluster, new_weights

    # Helper Functions (6): Calculate hinge distance
    def get_hinge_distance(
        self, spatial: torch.Tensor, e_spatial: torch.Tensor, y_cluster: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate hinge distance"""

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        return hinge, d

    # Helper Functions (7): Calculate truth from intersection between Prediction graph and Truth graph
    def get_truth(
        self, batch: Any, e_spatial: torch.Tensor, e_bidir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate truth from intersection between Prediction graph and Truth graph"""
        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        return e_spatial, y_cluster

    # Training Step
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step"""

        # Instantiate empty prediction edge list
        e_spatial = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        with torch.no_grad():
            spatial = self(input_data)

        query_indices, query = self.get_query_points(batch, spatial)

        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            e_spatial = self.append_hnm_pairs(e_spatial, query, query_indices, spatial)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            e_spatial = self.append_random_pairs(e_spatial, query_indices, spatial)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = torch.cat(
            [batch.signal_true_edges, batch.signal_true_edges.flip(0)], dim=-1
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        new_weights = y_cluster.to(self.device) * self.hparams["weight"]

        # Append all positive examples and their truth and weighting
        e_spatial, y_cluster, new_weights = self.get_true_pairs(
            e_spatial, y_cluster, new_weights, e_bidir
        )

        included_hits = e_spatial.unique()
        spatial[included_hits] = self(input_data[included_hits])
        # TODO: Choose between hinge loss, triplet loss, and cosine embedding loss

        # Calculate hinge distance
        hinge, d = self.get_hinge_distance(spatial, e_spatial, y_cluster)

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
        batch: Any,
        batch_idx: int,
        knn_radius: float,
        knn_num: int,
        log: bool = False,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Shared evaluation function for validation and test steps"""

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
            spatial, spatial, indices=None, r_max=knn_radius, k_max=knn_num
        )

        # Calculate truth from intersection between Prediction graph and Truth graph
        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)

        # Calculate hinge distance
        hinge, d = self.get_hinge_distance(
            spatial, e_spatial.to(self.device), y_cluster
        )

        # Calculate hinge loss
        loss = torch.nn.functional.hinge_embedding_loss(
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
            logging.info("Efficiency: {}".format(eff))
            logging.info("Purity: {}".format(pur))
            logging.info(batch.event_file)

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
    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Step to evaluate the model's performance"""
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_val"], 150, log=True
        )

        return outputs["loss"]

    # Test Step
    def test_step(self, batch: Data, batch_idx: int) -> Dict[str, Any]:
        """Step to evaluate the model's performance"""
        outputs = self.shared_evaluation(
            batch, batch_idx, self.hparams["r_test"], 1000, log=False
        )

        return outputs

    # Loss Functions: Hinge Loss, Triplet Loss, Cosine Embedding Loss, etc.
    def get_hinge_loss(
        self,
        spatial: torch.Tensor,
        e_spatial: torch.Tensor,
        y_cluster: torch.Tensor,
        new_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the hinge loss for the embedding model.

        This function computes the hinge loss, which is used to train the model to distinguish
        between positive pairs (points that should be close in the embedding space) and
        negative pairs (points that should be far apart).

        Args:
            spatial (torch.Tensor): The spatial embeddings of the points.
            e_spatial (torch.Tensor): Edge information, containing indices of point pairs.
            y_cluster (torch.Tensor): Binary labels for edges (0 for negative, 1 for positive pairs).
            new_weights (torch.Tensor): Weights for each edge pair.

        Returns:
            torch.Tensor: The total hinge loss, combining weighted positive and negative losses.

        The function performs the following steps:
        1. Calculates distances between point pairs and prepares hinge values.
        2. Applies a weight of 1 to negative examples.
        3. Computes separate losses for negative and positive pairs using hinge loss.
        4. Combines the negative and positive losses, with the positive loss weighted by a hyperparameter.

        The margin for the hinge loss is defined by the hyperparameter 'margin'.
        The weighting for positive examples is controlled by the hyperparameter 'weight'.
        """
        # Get the hinge distance between the reference and neighbor points
        hinge, d = self.get_hinge_distance(spatial, e_spatial, y_cluster)

        # Give negative examples a weight of 1 (note that there may still be TRUE examples that are weightless)
        new_weights[hinge == -1] = 1

        # Calculate the negative loss
        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=self.hparams["margin"] ** 2,
            reduction="mean",
        )

        # Calculate the positive loss
        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=self.hparams["margin"] ** 2,
            reduction="mean",
        )

        # Calculate and return the total loss
        return negative_loss + self.hparams["weight"] * positive_loss

    def get_triplet_loss(
        self,
        spatial: torch.Tensor,
        e_spatial: torch.Tensor,
        y_cluster: torch.Tensor,
        new_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the triplet loss for the embedding model.

        This function computes the triplet loss, which is used to train the model to minimize the
        distance between an anchor and a positive sample, while maximizing the distance between
        the anchor and a negative sample.

        Args:
            spatial (torch.Tensor): The spatial embeddings of the points.
            e_spatial (torch.Tensor): Edge information, containing indices of point triplets.
            y_cluster (torch.Tensor): Binary labels for edges (0 for negative, 1 for positive pairs).
            new_weights (torch.Tensor): Weights for each edge triplet.

        Returns:
            torch.Tensor: The total triplet loss.

        The function performs the following steps:
        1. Selects anchor, positive, and negative samples from the spatial embedding.
        2. Calculates distances between anchor-positive and anchor-negative pairs.
        3. Computes the triplet loss using these distances.
        4. Applies weights to the loss.

        The margin for the triplet loss is defined by the hyperparameter 'margin'.
        """
        # Select anchor, positive, and negative samples
        anchors = spatial.index_select(0, e_spatial[0])
        positives = spatial.index_select(0, e_spatial[1])
        negatives = spatial.index_select(0, e_spatial[2])

        # Calculate distances
        dist_pos = torch.sum((anchors - positives) ** 2, dim=-1)
        dist_neg = torch.sum((anchors - negatives) ** 2, dim=-1)

        # Compute triplet loss
        loss = torch.nn.functional.relu(dist_pos - dist_neg + self.hparams["margin"])

        # Apply weights
        weighted_loss = loss * new_weights

        # Return mean loss
        return weighted_loss.mean()

    def get_cosine_embedding_loss(
        self,
        spatial: torch.Tensor,
        e_spatial: torch.Tensor,
        y_cluster: torch.Tensor,
        new_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the cosine embedding loss for the embedding model.

        This function computes the cosine embedding loss, which uses cosine similarity
        instead of Euclidean distance. It's useful when the magnitude of embeddings
        is less important than their direction.

        Args:
            spatial (torch.Tensor): The spatial embeddings of all points.
            e_spatial (torch.Tensor): Edge information, containing indices of point pairs to compare.
            y_cluster (torch.Tensor): Binary labels for edges (0 for negative, 1 for positive pairs).
            new_weights (torch.Tensor): Weights for each edge pair.

        Returns:
            torch.Tensor: The total weighted cosine embedding loss.
        """
        # Select pairs of points
        x1 = spatial.index_select(0, e_spatial[0])
        x2 = spatial.index_select(0, e_spatial[1])

        # Convert binary labels to 1 and -1
        # Cosine embedding loss expects 1 for similar pairs and -1 for dissimilar pairs
        y = 2 * y_cluster.float() - 1

        # Compute cosine embedding loss
        # We use a default margin of 0.5, but this can be adjusted via hyperparameters
        loss = F.cosine_embedding_loss(
            x1, x2, y, margin=self.hparams.get("cosine_margin", 0.5), reduction="none"
        )

        # Apply weights
        weighted_loss = loss * new_weights

        # Return mean loss
        return weighted_loss.mean()
