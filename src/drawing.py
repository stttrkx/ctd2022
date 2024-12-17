#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .detector import detector_layout
from .math_utils import polar_to_cartesian

try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file (STT)
detector_path = os.path.join(output_base, "stt.csv")


def get_custom_cmap(color_list=None):
    """Get a custom color map using a list of colors similar to plt.get_cmap()
    function in matplotlib. See available color maps using plt.colormaps()"""

    if color_list is None:
        color_list = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "magenta",
            "black",
            "gray",
            "lime",
            "teal",
            "navy",
            "maroon",
            "olive",
            "indigo",
            "cyan",
        ]

    cmap = ListedColormap(color_list)
    return cmap


def Visualize_Edges(
    event,
    edges,
    figsize=(10, 10),
    fig_type="pdf",
    fig_name="event",
    save_fig=False,
    annotate=False,
):
    """Function to plot nodes and edges of an event."""

    # detector layout
    fig, ax = detector_layout(figsize=figsize)

    # get custom colormap for consistent track colors
    cmap = get_custom_cmap()

    # draw particles (event)
    unique_pids = np.unique(event.particle_id).astype(int)
    for pid in unique_pids:
        mask = event.particle_id == pid  # filter each pid
        color = cmap(pid % len(cmap.colors))  # cycle through cmap
        ax.scatter(
            event[mask].x.values,
            event[mask].y.values,
            color=color,
            label=f"particle_id: {pid}",
        )

    # draw edges (edges)
    # for i, (source_node, target_node) in enumerate(true_edges.T):
    for source_node, target_node in edges.T:

        # get source and target positions
        source_pos = event.loc[source_node, ["x", "y"]].values
        target_pos = event.loc[target_node, ["x", "y"]].values
        ax.plot(
            [source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], "k-", lw=0.5
        )

        if annotate:
            # annotate with an arrow
            ax.annotate(
                "",
                xy=(target_pos[0], target_pos[1]),
                xycoords="data",
                xytext=(source_pos[0], source_pos[1]),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", lw=0.5),
            )

    # axis params
    evtid = event.event_id.unique()[0]
    ax.set_title("Event # {}".format(evtid))
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()

    # save figure
    if save_fig:
        fig.savefig(f"{fig_name}_{evtid}.{fig_type}", format=fig_type)


# Draw SttCSVDataReader Event:: Using Object-Oriented API
def Visualize_CSVEvent(event=None, figsize=(10, 10), fig_type="pdf", save_fig=False):
    """Draw an event produced by SttCSVDataReader class."""

    # draw detector layout
    fig, ax = detector_layout(figsize=figsize)

    # get custom colormap for consistent track colors
    cmap = get_custom_cmap()

    # draw particles
    unique_pids = np.unique(event.particle_id.values).astype(int)
    for pid in unique_pids:
        df = event.loc[event.particle_id == pid]  # filter each pid
        color = cmap(pid % len(cmap.colors))  # cycle through cmap
        ax.scatter(df.x.values, df.y.values, color=color, label=f"particle_id: {pid}")

    # axis params
    evtid = np.unique(event.event_id.values)
    ax.set_title("Event # {}".format(evtid))
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()

    # save figure
    if save_fig:
        fig.savefig("event_{}.{}".format(evtid, fig_type))


# Draw SttTorchDataReader Event::  Using Object-Oriented API
def Visualize_TorchEvent(event=None, figsize=(10, 10), fig_type="pdf", save_fig=False):
    """Draw an event produced by SttTorchDataReader class."""

    # draw detector layout
    fig, ax = detector_layout(figsize=figsize)

    # get custom colormap for consistent track colors
    cmap = get_custom_cmap()

    # feature data
    x, y = polar_to_cartesian(r=event.x[:, 0], phi=event.x[:, 1])

    # draw particles
    unique_pids = np.unique(event.pid).astype(int)
    for pid in unique_pids:
        mask = event.pid == pid  # filter each pid
        color = cmap(pid % len(cmap.colors))  # cycle through cmap
        ax.scatter(x[mask], y[mask], color=color, label=f"particle_id: {pid}")

    # axis params
    evtid = int(event.event_file[-10:])
    ax.set_title("Event # {}".format(evtid))
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()

    # save figure
    if save_fig:
        fig.savefig("event_{}.{}".format(evtid, fig_type))
