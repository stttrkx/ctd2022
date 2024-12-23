#!/usr/bin/env python
# coding: utf-8

"""Heuristic Methods for Graph Construction."""

import logging
import itertools
import scipy as sp
import numpy as np
import pandas as pd
import torch

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Create an edge between every hit pair
def get_all_edges(hits):
    """Create an edge between every hit pair.

    Args:
        hits (_type_): Pandas DataFrame containing hit information.

    Returns:
        _type_: A 2D numpy array of shape (2, n_edges) where each column is an edge between two hits.
    """

    nHits = hits.shape[0]
    edgeStart = []
    edgeEnd = []

    for i in range(nHits - 1):
        for j in range(i + 1, nHits):
            edgeStart.append(i)
            edgeEnd.append(j)

    input_graph = np.array([edgeStart, edgeEnd])
    return input_graph


# Construction Graph based on a Heuristic
def construct_graphs(hits, hm="layerwise_hm", adjacent_sectors=True, directional=True):
    """Top-level method to construct graph usig a heuristic method"""

    if hm == "layerwise_hm":
        input_edges = construct_layerwise_edges(hits, adjacent_sectors=True)
    elif hm == "modulewise_hm":
        input_edges = construct_modulewise_graph(
            hits, adjacent_sectors=True, directional=True
        )
    else:
        logging.error(f"{hm} is not a valid method to build input graphs.")
        exit(1)

    # concatenate and transform
    input_edges = pd.concat(input_edges, axis=0).to_numpy().T
    return input_edges


# Layerwise Heuristic Method:
def construct_layerwise_edges(hits, adjacent_sectors=True):
    """
    Construct input edges (graph) in adjacent layers & sectors of the detector.

    Parameters
    ----------
    hits : pd.DataFrame
        hits information
    sector_mask : bool, optional
        if True, edge construction constrained in adjacent sectors
        if False, do not require any sector constraint
    """

    # construct layer pairs
    layers = hits.layer_id.unique()
    layer_pairs = np.stack([layers[:-1], layers[1:]], axis=1)

    # layerwise construction
    layerwise_edges = []
    layer_groups = hits.groupby("layer_id")
    for l1, l2 in layer_pairs:

        # get l1, l2 layer groups
        try:
            lg1 = layer_groups.get_group(l1)
            lg2 = layer_groups.get_group(l2)
        # If an event has no hits on a layer, we get a KeyError.
        # In that case we just skip to the next layer pair.
        except KeyError as e:
            logging.info("skipping empty layer: %s" % e)
            continue

        # get all possible pairs of hits
        keys = ["event_id", "r", "phi", "isochrone", "sector_id"]
        hit_pairs = (
            lg1[keys]
            .reset_index()
            .merge(lg2[keys].reset_index(), on="event_id", suffixes=("_1", "_2"))
        )

        # construct edges with/without sector constraint
        if adjacent_sectors:
            dSector = hit_pairs["sector_id_1"] - hit_pairs["sector_id_2"]
            sector_mask = (dSector.abs() < 2) | (dSector.abs() == 5)
            segments = hit_pairs[["index_1", "index_2"]][sector_mask]
        else:
            segments = hit_pairs[["index_1", "index_2"]]

        # append edge list
        layerwise_edges.append(segments)

    return layerwise_edges


# Modulewise Heuristic Method:
def construct_samelayer_edges(hits, directional=True):
    """
    Construct input edges (graph) between adjacent hits in the same layer of the detector.

    Parameters
    ----------
    hits : pd.DataFrame
        hits information of an event
    directional : bool, optional
        if True, directional edge with no self-loop
        if False, bidirectional edge with no self-loop
    """

    # construct samelayer edges
    samelayer_edges = []
    layer_groups = hits.groupby("layer_id")
    for _, lg in layer_groups:
        if len(lg) > 1:

            # same keys as above
            keys = ["event_id", "r", "phi", "isochrone", "sector_id"]
            pairs = (
                lg[keys]
                .reset_index()
                .merge(lg[keys].reset_index(), on="event_id", suffixes=("_1", "_2"))
            )

            # Remove self-loops
            if directional:
                pairs = pairs[pairs["index_1"] < pairs["index_2"]]  # directional edge
            else:
                pairs = pairs[
                    pairs["index_1"] != pairs["index_2"]
                ]  # bidirectional edge

            # Collect pairs for edges
            samelayer_edges.append(pairs[["index_1", "index_2"]])

    return samelayer_edges


def construct_modulewise_graph(hits, adjacent_sectors=True, directional=True):
    """
    Get input graph (edges) in adjacent layers and sectors with or without inner edges.

    Parameters
    ----------
    hits : pd.DataFrame
        hits information
    sector_mask : bool, optional
        if True, edge construction in adjacent sectors
        if False, do not require any sector constraint
    inneredges : bool, optional
        if True, include edges within a layer (samelayer edges)
        if False, do not include samelayer edges
    directional : bool, optional
        if True, excludes self-loops on samelayer edges
        if False, includes self-loops on samelayer edges
    """

    # construct graph
    layerwise_edges = construct_layerwise_edges(hits, adjacent_sectors)
    samelayer_edges = construct_samelayer_edges(hits, directional)
    modulewise_edges = itertools.chain(layerwise_edges, samelayer_edges)

    return modulewise_edges


# Graph Intersection to build Labelled Dataset ([edge_index, y])
def graph_intersection(pred_graph, truth_graph):
    """Graph Intersection to build Labelled Dataset ([edge_index, y])

    Parameters
    ----------
    pred_graph : torch.tensor or numpy.array
        input graph e.g. layerwise_input_graph

    truth_graph : torch.tensor or numpy.array
        true graph e.g. layerwise_true_edges
    """

    # get the size of the largest graph
    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    # convert to numpy array if tensor
    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph

    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph

    # create sparse matrices
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()

    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()

    # delete temporary variables
    del l1
    del l2

    # compute the intersection of the two graphs
    e_intersection = (e_1.multiply(e_2) - ((e_1 - e_2) > 0)).tocoo()

    # delete temporary variables
    del e_1
    del e_2

    # create labelled dataset PyG: [edge_index, y]
    new_pred_graph = (
        torch.from_numpy(np.vstack([e_intersection.row, e_intersection.col]))
        .long()
        .to(device)
    )
    y = torch.from_numpy(e_intersection.data > 0).to(device)

    # delete temporary variables
    del e_intersection

    return new_pred_graph, y
