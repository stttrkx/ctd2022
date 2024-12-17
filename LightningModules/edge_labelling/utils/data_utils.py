#!/usr/bin/env python
# coding: utf-8

import os
import random
import torch
from torch.utils.data import random_split

# Find the current device.
device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: Fetch events from 'feature_store', shuffle & split according to
# 'datatype_split' variable provided by the train_quickstart_GNN.yaml


def split_datasets(
    input_dir="",
    train_split=None,
    pt_background_cut=0,
    pt_signal_cut=0,
    noise=True,
    seed=1,
    **kwargs
):
    """
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should
    be changed across final varied runs, but can be left as default for experimentation.
    """
    print("Loading data to '{}'".format(device))

    # random seed
    if train_split is None:
        train_split = kwargs["datatype_split"]
    torch.manual_seed(seed)

    # load data
    loaded_events = load_dataset(
        input_dir, sum(train_split), pt_background_cut, pt_signal_cut, noise, **kwargs
    )

    # split data
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    print(
        "Trainset: {}, Valset: {}, Testset: {}\n".format(
            len(train_events), len(val_events), len(test_events)
        )
    )

    return train_events, val_events, test_events


def load_dataset(
    input_subdir="",
    num_events=10,
    pt_background_cut=0,
    pt_signal_cut=0,
    noise=False,
    **kwargs
):

    # Load dataset from a subdir
    if input_subdir is not None:
        all_events = os.listdir(input_subdir)

        if "sorted_events" in kwargs.keys() and kwargs["sorted_events"]:
            all_events = sorted(all_events)
        else:
            random.shuffle(all_events)

        all_events = [os.path.join(input_subdir, event) for event in all_events]
        loaded_events = [
            torch.load(event, map_location=torch.device("cpu"))
            for event in all_events[:num_events]
        ]

        loaded_events = select_data(
            loaded_events, pt_background_cut, pt_signal_cut, noise
        )
        return loaded_events
    else:
        return None


def select_data(events, pt_background_cut, pt_signal_cut, noise):

    # Handle event in batched form
    if type(events) is not list:
        events = [events]

    # NOTE: Cutting background by pT BY DEFINITION removes noise
    if (pt_background_cut > 0) or not noise:
        for event in events:

            edge_mask = (
                (event.pt[event.edge_index] > pt_background_cut)
                & (event.pid[event.edge_index] == event.pid[event.edge_index])
                & (event.pid[event.edge_index] != 0)
            ).all(0)

            # Apply Mask on "edge_index, y, weights, y_pid"
            event.edge_index = event.edge_index[:, edge_mask]

            if "y" in event.__dict__.keys():
                event.y = event.y[edge_mask]

            if "weights" in event.__dict__.keys():
                if event.weights.shape[0] == edge_mask.shape[0]:
                    event.weights = event.weights[edge_mask]

            if "y_pid" in event.__dict__.keys():
                event.y_pid = event.y_pid[edge_mask]

    for event in events:
        if "y_pid" not in event.__dict__.keys():
            event.y_pid = (
                event.pid[event.edge_index[0]] == event.pid[event.edge_index[1]]
            ) & event.pid[event.edge_index[0]].bool()

        if (
            "signal_true_edges" in event.__dict__.keys()
            and event.signal_true_edges is not None
        ):
            signal_mask = (event.pt[event.signal_true_edges] > pt_signal_cut).all(0)
            event.signal_true_edges = event.signal_true_edges[:, signal_mask]

    return events
