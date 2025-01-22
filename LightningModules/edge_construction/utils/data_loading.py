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

# FIXME: I want to move data loading and preprocess (need for embedding only)
# utils here. Once done, fix paths and duplicates from embedding_utils.py.


# ------------------------------- Data Handling --------------------------------
def split_datasets(
    input_dir="",
    train_split=None,
    pt_background_cut=0,
    pt_signal_cut=0,
    min_nhits=0,
    primary_only=False,
    true_edges=None,
    noise=True,
    seed=1,
    **kwargs
):
    """
    Prepare random train, val, and test split, using a seed for reproducibility.
    Seed should be changed across final varied runs, but can be left as default
    for experimentation.
    """

    torch.manual_seed(seed)

    # Handle Data Split
    if train_split is None:
        train_split = [100, 10, 10]

    # Load Dataset
    loaded_events = load_dataset(
        input_dir,
        sum(train_split),
        pt_background_cut,
        pt_signal_cut,
        min_nhits,
        primary_only,
        true_edges,
        noise,
        **kwargs
    )

    # Split Dataset
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events


def load_dataset(
    input_dir,
    n_events,
    pt_background_cut,
    pt_signal_cut,
    min_nhits,
    primary_only,
    true_edges,
    noise,
    **kwargs
):
    """Load events from an input directory upto a number."""

    if input_dir is not None:
        # list all events
        all_events = os.listdir(input_dir)
        all_events = sorted([os.path.join(input_dir, event) for event in all_events])

        # load events needed
        loaded_events = []
        for event in all_events[:n_events]:
            try:
                loaded_event = torch.load(event, map_location=torch.device("cpu"))
                loaded_events.append(loaded_event)
            except:
                logging.info("Corrupted event file: {}".format(event))

        # apply selection on data
        selected_events = select_data(
            loaded_events,
            pt_background_cut,
            pt_signal_cut,
            min_nhits,
            primary_only,
            true_edges,
            noise,
        )

        return selected_events
    else:
        return None


def select_data(
    events, pt_background_cut, pt_signal_cut, min_nhits, primary_only, true_edges, noise
):
    """Select data fields, apply selection cuts and add new data fields."""

    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if pt_background_cut > 0 or not noise:
        for event in events:

            # get pt_mask
            pt_mask = (
                (event.pt > pt_background_cut)
                & (event.pid == event.pid)
                & (event.pid != 0)
            )

            pt_where = torch.where(pt_mask)[0]

            inverse_mask = torch.zeros(pt_where.max() + 1).long()
            inverse_mask[pt_where] = torch.arange(len(pt_where))

            event[true_edges], edge_mask = get_edge_subset(
                event[true_edges], pt_where, inverse_mask
            )

            # apply pt_mask on node features
            node_features = ["x", "hid", "pid", "pt", "nhits", "primary"]
            for feature in node_features:
                if feature in event.keys:
                    event[feature] = event[feature][pt_mask]

    # Filter events based on signal pt, primary particles, nhits
    for event in events:

        event.signal_true_edges = event[true_edges]
        edge_subset = torch.ones(event.signal_true_edges.shape[1]).bool()

        if "pt" in event.keys:
            edge_subset &= (event.pt[event[true_edges]] > pt_signal_cut).all(0)

        if "primary" in event.keys:
            edge_subset &= (event.nhits[event[true_edges]] >= min_nhits).all(0)

        if "nhits" in event.keys:
            edge_subset &= event.primary[event[true_edges]].bool().all(0) | (
                not primary_only
            )

        # get final true edges
        event.signal_true_edges = event.signal_true_edges[:, edge_subset]

    return events


def get_edge_subset(edges, mask_where, inverse_mask):
    """Filter edges based on a mask and also return the final edge mask."""
    included_edges_mask = np.isin(edges, mask_where).all(0)
    included_edges = edges[:, included_edges_mask]
    included_edges = inverse_mask[included_edges]

    return included_edges, included_edges_mask


def reset_edge_id(subset, graph):
    subset_ind = np.where(subset)[0]
    filler = -np.ones((graph.max() + 1,))
    filler[subset_ind] = np.arange(len(subset_ind))
    graph = torch.from_numpy(filler[graph]).long()
    exist_edges = (graph[0] >= 0) & (graph[1] >= 0)
    graph = graph[:, exist_edges]

    return graph, exist_edges
