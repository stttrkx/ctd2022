#!/usr/bin/env python
# coding: utf-8

"""Embedding loss functions e.g. (weighted) hinge, triplet & cosine losses.
This files is not intended to import directly, but to be copied into other"""

import os
import logging
import torch
from torch.utils.data import random_split
from torch import nn
import scipy as sp
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


# Main steering function for calculatig loss.
def loss_function(spatial, e_spatial, y_cluster, weights=None, loss_type="hinge"):
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

    # Compute squared Euclidean distances b/w pairs of points
    d_sq = self.squared_distances(spatial, e_spatial)

    # Handle edge weights (default: 2.0)
    weights = torch.ones_like(d_sq) if weights is None else weights

    # Convert actual labels (0, 1) to hinge labels (-1, +1)
    hinge = 2 * y_cluster.float().to(self.device) - 1

    return self.get_weighted_hinge_loss_new(hinge, d_sq, weights)


# Squared Euclidean distances between pairs.
def squared_distances(spatial, e_spatial):
    """Compute squared Euclidean distances between pairs of points.
    Args:
        spatial (torch.Tensor): Node embeddings of shape (num_nodes, embedding_dim).
        e_spatial (torch.Tensor): Edge indices of shape (2, num_edges).
    Returns:
        d (torch.Tensor): Squared distances between reference and neighbour nodes of shape (num_edges,).
    """
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

    return d_sq


# ------------------------ Weighted Hinge Loss
def get_weighted_hinge_loss_new(hinge, d, weights):
    """Compute the weighted hinge loss for the embedding model. It uses "none" reduction in
    hinge loss calculation, requiring manual mean calculation. However, it only applies
    weight to similar edges. No way to adjust weights for dissimilar edges (disadvantage).
    Args:
        hinge (torch.Tensor): Hinge labels (1 for similar, -1 for dissimilar) of shape (num_edges,).
        d (torch.Tensor): Squared distances between reference and neighbour nodes of shape (num_edges,).
        weights (torch.Tensor): Weights for each edge of shape (num_edges,).
    Returns:
        torch.Tensor: The total weighted hinge loss.
    """

    # Hangle hinge margin (default: 0.1)
    hinge_margin = self.hparams.get("margin", 0.1) ** 2

    # Essentially the same as get_weighted_hinge_loss(), it simply uses a weight
    # tensor instead of a scalar weight parameter. One way to calculate weighted
    # loss is to apply weights early where reduction="mean" become possible. This
    # can be achived by first weighting the distance, then calculating the loss:
    # d = d * new_weights  # early weighting to use "mean" reduction
    # weighted_loss = torch.nn.functional.hinge_embedding_loss(
    #    d, hinge, margin=hinge_margin, reduction="mean"
    # )  # gives same loss calculation as given below.

    # Negative loss: Push dissimilar pairs apart (hinge == -1)
    negative_mask = hinge == -1
    negative_loss = torch.nn.functional.hinge_embedding_loss(
        d[negative_mask],
        hinge[negative_mask],
        margin=hinge_margin,
        reduction="none",
    )
    negative_loss = (negative_loss * weights[negative_mask]).mean()

    # Positive loss: Pull similar pairs closer (hinge == 1)
    positive_mask = hinge == 1
    positive_loss = torch.nn.functional.hinge_embedding_loss(
        d[positive_mask],
        hinge[positive_mask],
        margin=hinge_margin,
        reduction="none",
    )
    positive_loss = (positive_loss * weights[positive_mask]).mean()

    # Weighted hinge loss
    weighted_loss = negative_loss + positive_loss

    return weighted_loss


# gnn4itk-like loss, simple weight handling given by external 'weight' param
def weighted_hinge_loss_gnn4stt(y_cluster, d, weights):
    """Compute the weighted hinge loss for an embedding model. It uses
    actual edge labels (0, 1) with hinge labels (-1, 1) created on the fly.

    Args:
        y_cluster (torch.Tensor): Edge labels (1 for similar, 0 for dissimilar) of shape (num_edges,).
        d (torch.Tensor): Squared distances between reference and neighbour nodes of shape (num_edges,).
        weights (torch.Tensor): Weights for each edge of shape (num_edges,).
            Positive weights maintain relationship, negative weights flip it, zero weights exclude pairs.
    Returns:
        torch.Tensor: The total weighted hinge loss.
    """

    # FIXME: To make it similar to weighted_hinge_loss_gnn4itk(), understand
    # weighting scheme used. We've' weight given by self.hparams["weight"].

    # Hangle hinge margin (default: 0.1)
    hinge_margin = self.hparams.get("margin", 0.1) ** 2

    # Negative mask to handle weight signs and zeros
    negative_mask = ((y_cluster == 0) & (weights > 0)) | (
        (y_cluster == 1) & (weights < 0)
    )
    negative_mask = negative_mask & (weights != 0)

    # Negative loss: Push dissimilar pairs apart
    # negative_mask = ((y_cluster == 0) & (weights != 0)) | (weights < 0)
    negative_loss = torch.nn.functional.hinge_embedding_loss(
        d[negative_mask],
        torch.ones_like(d[negative_mask]) * -1,  # Hinge label is always -1
        margin=hinge_margin,
        reduction="none",
    )
    # Use absolute weights
    negative_loss = (negative_loss * weights[negative_mask].abs()).mean()

    # Positive mask to handle weight signs and zeros
    positive_mask = ((y_cluster == 1) & (weights > 0)) | (
        (y_cluster == 0) & (weights < 0)
    )
    positive_mask = positive_mask & (weights != 0)

    # Positive loss: Pull similar pairs closer
    positive_loss = torch.nn.functional.hinge_embedding_loss(
        d[positive_mask],
        torch.ones_like(d[positive_mask]),  # Hinge label is always +1
        margin=hinge_margin,
        reduction="none",
    )
    # Use absolute weights
    positive_loss = (positive_loss * weights[positive_mask].abs()).mean()

    # Weighted hinge loss
    weighted_loss = negative_loss + positive_loss

    return weighted_loss


# gnn4itk loss, complex weight handling given by external ''weight_spec' param
def weighted_hinge_loss_gnn4itk(y_cluster, d, weights):
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

    # Hangle hinge margin (default: 0.1)
    hinge_margin = self.hparams.get("margin", 0.1) ** 2

    # Negative mask to push dissimilar pairs apart
    negative_mask = ((y_cluster == 0) & (weights != 0)) | (weights < 0)

    # Handle negative loss, but don't reduce vector
    negative_loss = torch.nn.functional.hinge_embedding_loss(
        d[negative_mask],
        torch.ones_like(d[negative_mask]) * -1,
        margin=hinge_margin,
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
        margin=hinge_margin,
        reduction="none",
    )
    # Now reduce the vector with non-zero weights
    positive_loss = (positive_loss * weights[positive_mask].abs()).mean()

    # Weighted hinge loss
    weighted_loss = negative_loss + positive_loss

    return weighted_loss


# ------------------------ Weighted Triplet Loss
def weighted_triplet_loss(
    self,
    spatial: torch.Tensor,
    e_spatial: torch.Tensor,
    y_cluster: torch.Tensor,
    new_weights: torch.Tensor,
) -> torch.Tensor:

    # Hangle hinge margin (default: 0.1)
    triplet_margin = self.hparams.get("triplet_margin", 1.0)

    # Determine anchors and neighbors in embedding space
    anchors = spatial[e_spatial[0]]
    neighbors = spatial[e_spatial[1]]

    # Determine positive and negative samples based on y_cluster
    positives = neighbors[y_cluster == 1]
    negatives = neighbors[y_cluster == 0]

    # Ensure we have equal number of positives and negatives
    min_samples = min(positives.shape[0], negatives.shape[0])
    if min_samples == 0:
        return torch.tensor(0.0, device=self.device)  # No valid triplets in this batch

    anchors = anchors[:min_samples]
    positives = positives[:min_samples]
    negatives = negatives[:min_samples]

    # Compute triplet loss
    loss = torch.nn.functional.triplet_margin_loss(
        anchors,
        positives,
        negatives,
        margin=triplet_margin,
        reduction="none",
    )

    # Apply weights (we need to adjust weights to match the reduced sample size)
    relevant_weights = new_weights[:min_samples]
    weighted_loss = loss * relevant_weights

    return weighted_loss.mean()


# ------------------------ Weighted Cosine Loss
