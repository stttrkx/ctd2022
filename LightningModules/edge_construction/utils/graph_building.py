#!/usr/bin/env python
# coding: utf-8

"""Ideally, we would be using FRNN and the GPU. But in the case of a user not having
a GPU, or not having FRNN, we import FAISS as the nearest neighbor library """

import os
import logging
import numpy as np
import scipy as sp
import torch
from torch import nn
from torch.utils.data import random_split
import faiss
import faiss.contrib.torch_utils

try:
    import frnn

    FRNN_AVAILABLE = True
except ImportError:
    FRNN_AVAILABLE = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    FRNN_AVAILABLE = False

FRNN_AVAILABLE = False

# FIXME: I want to move graph building, graph intersection utils here. Once done
# fix relavant paths and remove duplicate functions from embedding_utils.py.


# ------------------------------- Graph Building -------------------------------
def build_edges(
    query, database, indices=None, r_max=1.0, k_max=10, return_indices=False
):
    """NOTE: The KNN/FRNN algorithms return the distances**2. Therefore, we need
    to be careful when comparing them to the target distances (r_val, r_test),
    and to the margin parameter (which is L1 distance).
    """

    # use FRNN library
    if FRNN_AVAILABLE:

        Dsq, I, nn, grid = frnn.frnn_grid_points(
            points1=query.unsqueeze(0),
            points2=database.unsqueeze(0),
            lengths1=None,
            lengths2=None,
            K=k_max,
            r=r_max,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )

        I = I.squeeze().int()
        ind = torch.Tensor.repeat(
            torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
        ).T.int()

        positive_idxs = I >= 0
        edge_list = torch.stack([ind[positive_idxs], I[positive_idxs]]).long()

    # use FAISS library
    else:

        if device == "cuda":
            res = faiss.StandardGpuResources()
            Dsq, I = faiss.knn_gpu(res=res, xq=query, xb=database, k=k_max)
        elif device == "cpu":
            index = faiss.IndexFlatL2(database.shape[1])
            index.add(database)
            Dsq, I = index.search(query, k_max)

        ind = torch.Tensor.repeat(
            torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
        ).T.int()

        edge_list = torch.stack([ind[Dsq <= r_max**2], I[Dsq <= r_max**2]])

    # Reset indices subset to correct global index
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    if return_indices:
        return edge_list, Dsq, I, ind
    else:
        return edge_list


def build_knn(spatial, k_max):
    """Build edges using kNN algorithm from FAISS library. One can also use
    Scikit-learn for this purpose but its not optimized for GPUs."""

    if device == "cuda":
        res = faiss.StandardGpuResources()
        _, I = faiss.knn_gpu(res=res, xq=spatial, xb=spatial, k=k_max)
    elif device == "cpu":
        index = faiss.IndexFlatL2(spatial.shape[1])
        index.add(spatial)
        _, I = index.search(spatial, k_max)

    ind = torch.Tensor.repeat(
        torch.arange(I.shape[0], device=device), (I.shape[1], 1), 1
    ).T
    edge_list = torch.stack([ind, I])

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    return edge_list


# ----------------------------- Graph Intersection -----------------------------
def graph_intersection(
    pred_graph, truth_graph, using_weights=False, weights_bidir=None
):
    """Graph Intersection to build Labelled Dataset ([edge_index, y])"""

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph

    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph

    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()

    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()

    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()

        del weights_list
        del l2

        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)

    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)

    del e_intersection

    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y
