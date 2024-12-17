#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    raise e


# detector geometry file
detector_path = os.path.join(output_base, "stt.csv")


# Draw STT: Using Object Oriented API
def detector_layout(figsize=(10, 10)):
    """Draw detector layout (approx.), intended as base for further plotting"""

    plt.close("all")

    # init subplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # detector layout
    data = pd.read_csv(detector_path)
    data["sign"] = np.sign(data["angle"])

    # filter tubes
    parallel_tubes = data.query("sign == 0")  # Parallel tubes
    pos_skewed_tubes = data.query("sign == 1")  # Positively skewed tubes
    neg_skewed_tubes = data.query("sign == -1")  # Negatively skewed tubes

    # draw tubes
    # ax.scatter(data.x.values, data.y.values, s=0.5, facecolors='none', edgecolors='lightgrey')
    ax.scatter(
        parallel_tubes.x.values,
        parallel_tubes.y.values,
        s=50,
        facecolors="none",
        edgecolors="lightgreen",
    )  # parallel
    ax.scatter(
        pos_skewed_tubes.x.values,
        pos_skewed_tubes.y.values,
        s=50,
        facecolors="none",
        edgecolors="royalblue",
    )  # positive skewed
    ax.scatter(
        neg_skewed_tubes.x.values,
        neg_skewed_tubes.y.values,
        s=50,
        facecolors="none",
        edgecolors="lightcoral",
    )  # negative skewed

    # plotting params
    ax.set_xlabel("x [cm]", fontsize=15)
    ax.set_ylabel("y [cm]", fontsize=15)
    ax.set_xlim(-41, 41)
    ax.set_ylim(-41, 41)
    ax.set_aspect("equal")
    ax.grid(False)

    return fig, ax


def detector_layout_new(figsize=(10, 10)):
    """Draw detector layout (approx.), intended as base for further plotting"""

    plt.close("all")

    # init subplots
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # detector layout
    data = pd.read_csv(detector_path)
    data["sign"] = np.sign(data["angle"])

    # draw tubes
    for index, row in data.iterrows():
        if row["sign"] == 0:  # Parallel tubes
            straightOuterTube = Circle(
                (row["x"], row["y"]), row["outer_radius"], fc="None", ec="lightgreen"
            )
            ax.add_patch(straightOuterTube)
            straightInnerTube = Circle(
                (row["x"], row["y"]), row["inner_radius"], fc="None", ec="lightgreen"
            )
            ax.add_patch(straightInnerTube)
        elif row["sign"] == 1:  # Positively skewed tubes
            posSkewedOuterTube = Circle(
                (row["x"], row["y"]), row["outer_radius"], fc="None", ec="royalblue"
            )
            ax.add_patch(posSkewedOuterTube)
            posSkewedInnerTube = Circle(
                (row["x"], row["y"]), row["inner_radius"], fc="None", ec="royalblue"
            )
            ax.add_patch(posSkewedInnerTube)
        elif row["sign"] == -1:  # Negatively skewed tubes
            negSkewedOuterTube = Circle(
                (row["x"], row["y"]), row["outer_radius"], fc="None", ec="lightcoral"
            )
            ax.add_patch(negSkewedOuterTube)
            negSkewedInnerTube = Circle(
                (row["x"], row["y"]), row["inner_radius"], fc="None", ec="lightcoral"
            )
            ax.add_patch(negSkewedInnerTube)
        else:
            raise Exception("Invalid angle")

    # plotting params
    ax.set_xlabel("x [cm]", fontsize=15)
    ax.set_ylabel("y [cm]", fontsize=15)
    ax.set_xlim(-41, 41)
    ax.set_ylim(-41, 41)
    ax.set_aspect("equal")
    ax.grid(False)

    return fig, ax
