#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import random
import numpy as np

# Find current device.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Only import cupy in CUDA environment.
if device == "cuda":
    import cupy as cp


# ---------------------- Dense Network
def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="None",
    layer_norm=False,
    batch_norm=False,
):
    """Construct an MLP with Specified Fully-connected Layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)

    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes

    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))

        # LayerNorm & BatchNorm
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[i + 1], affine=False, track_running_stats=False)
            )

        layers.append(hidden_activation())

    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:

        # LayerNorm & BatchNorm
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[-1], affine=False, track_running_stats=False)
            )
        layers.append(output_activation())

    return nn.Sequential(*layers)


# ---------------------- Data Processing
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
