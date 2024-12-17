#!/usr/bin/env python

"""Plotting utilities"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import itertools
import atlasify as atl
from atlasify import atlasify

fontsize = 18
minor_size = 14

# pt params
# pt_bins = (np.arange(0, 5., step=0.5).tolist() + np.arange(5, 11, step=1.0).tolist())
# pt_bins = np.arange(0.1, 1.5, step=0.1) # prob: last value isn't included
pt_bins = np.linspace(0.1, 1.5, num=15)  # will give 15 bins

pt_configs = {"bins": pt_bins, "histtype": "step", "lw": 2, "log": False}

# eta params
eta_bins = np.arange(-4, 4.4, step=0.4)

eta_configs = {"bins": eta_bins, "histtype": "step", "lw": 2, "log": False}


# get a plot
def get_plot(nrows=1, ncols=1, figsize=(8, 6), nominor=False):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)

    def format_axis(axis):
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        return axis

    if nrows * ncols == 1:
        ax = axs
        if not nominor:
            format_axis(ax)
    else:
        ax = [format_axis(x) if not nominor else x for x in axs.flatten()]

    return fig, ax


# add up x-axis
def add_up_xaxis(ax):
    """creating a top x-axis that mirrors the bottom x-axis in terms of tick positions, but without any labels."""
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(["" for x in ax.get_xticks()])
    ax2.xaxis.set_minor_locator(AutoMinorLocator())


# Get ratio
def get_ratio(x_vals, y_vals):
    res = [x / y if y != 0 else 0.0 for x, y in zip(x_vals, y_vals)]
    err = [
        x / y * math.sqrt((x + y) / (x * y)) if y != 0 and x != 0 else 0.0
        for x, y in zip(x_vals, y_vals)
    ]
    return res, err


# Get pairwise
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# Add mean and std
def add_mean_std(
    array, x, y, ax, color="k", dy=0.3, digits=2, fontsize=12, with_std=True
):
    this_mean, this_std = np.mean(array), np.std(array)
    ax.text(
        x,
        y,
        "Mean: {0:.{1}f}".format(this_mean, digits),
        color=color,
        fontsize=fontsize,
    )
    if with_std:
        ax.text(
            x,
            y - dy,
            "Standard Deviation: {0:.{1}f}".format(this_std, digits),
            color=color,
            fontsize=12,
        )


# Make Plot
def make_cmp_plot(
    arrays,
    configs,
    xlabel,
    ylabel,
    legends,
    loc,
    ymin,
    ymax,
    ratio_ylabel,
    ratio_legends,
    ratio_loc,
    badgetext,
    ref_lines,
    figsize,
    outname,
):

    bins = 0.0
    vals_list = []
    COLORS = ["blue", "blue", "red", "red"]
    LINESTYLES = ["solid", "dashed", "solid", "dashed"]

    # Plot 1: make a particle plot
    fig, ax = get_plot(nrows=1, ncols=1, figsize=figsize)
    for i, (array, legend) in enumerate(zip(arrays, legends)):
        vals, bins, _ = ax.hist(
            array, **configs, color=COLORS[i], label=legend, linestyle=LINESTYLES[i]
        )
        vals_list.append(vals)

    # Axes Params
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=minor_size)
    ax.tick_params(axis="both", which="minor", labelsize=minor_size)
    # ax.set_ylim([0., 1.])
    # add_up_xaxis(ax)
    ax.legend(loc=loc, fontsize=14, handlelength=1.5, handleheight=0.5)
    ax.grid(False)

    # Fig Badge (ATLAS, BELLE II, PANDA, etc)
    # atl.ATLAS = "PANDA"
    # atl.monkeypatch_axis_labels()
    atl.atlasify(
        "Internal",
        badgetext,  # Subtext
        font_size=18,  # Main Badge
        label_font_size=18,  # Text After Badge
        sub_font_size=14,  # Subtext
        subtext_distance=0.1,
        brand="PANDA",
        # enlarge=1.2,
        # outside=True
    )

    # Fig Params
    fig.savefig(f"{outname}.pdf", dpi=300)

    # Plot 2: make a ratio plot
    fig, ax = get_plot(nrows=1, ncols=1, figsize=figsize)
    xvals = [0.5 * (x[1] + x[0]) for x in pairwise(bins)]
    xerrs = [0.5 * (x[1] - x[0]) for x in pairwise(bins)]

    # Physics Efficiency:
    ratio, ratio_err = get_ratio(vals_list[1], vals_list[0])
    label = None if ratio_legends is None else ratio_legends[0]
    ax.errorbar(
        xvals,
        ratio,
        yerr=ratio_err,
        fmt="o",
        xerr=xerrs,
        lw=1,
        color="blue",
        label=label,
    )
    # print("Physics Efficiency: ", ratio)

    # Technical Efficiency:
    ratio, ratio_err = get_ratio(vals_list[3], vals_list[2])
    label = None if ratio_legends is None else ratio_legends[1]
    ax.errorbar(
        xvals,
        ratio,
        yerr=ratio_err,
        fmt="o",
        xerr=xerrs,
        lw=1,
        color="red",
        label=label,
    )
    # print("\nTechnical Efficiency: ", ratio)

    # Ref. Lines for pt Plots:
    if ref_lines:
        ax.axhline(y=0.90, xmin=0.0, xmax=1.0, color="black", linestyle="--")
        ax.axvline(x=0.25, ymin=0.0, ymax=0.77, color="green", linestyle="-.")

    # Axes Params
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ratio_ylabel, fontsize=fontsize)
    ax.set_ylim([0.0, 1.0])
    ax.tick_params(axis="both", which="major", labelsize=minor_size)
    ax.tick_params(axis="both", which="minor", labelsize=minor_size)
    # add_up_xaxis(ax)
    ax.legend(loc=ratio_loc, fontsize=14)
    ax.grid(False)

    # Fig Badge (ATLAS, BELLE II, PANDA, etc)
    # atl.ATLAS = "PANDA"
    # atl.monkeypatch_axis_labels()
    atl.atlasify(
        "Internal",
        badgetext,  # Subtext
        font_size=18,  # Main Badge
        label_font_size=18,  # Text After Badge
        sub_font_size=14,  # Subtext
        subtext_distance=0.1,
        brand="PANDA",
        # enlarge=1.2,
        # outside=True
    )

    # Fig Params
    fig.savefig(f"{outname}_ratio.pdf", dpi=300)


# Make Plot
def make_cmp_plot_old(
    arrays, legends, configs, xlabel, ylabel, ratio_label, ratio_legends, outname, ymin
):

    bins = 0.0
    vals_list = []

    # make a particle plot
    fig, ax = get_plot()
    for array, legend in zip(arrays, legends):
        vals, bins, _ = ax.hist(array, **configs, label=legend)
        vals_list.append(vals)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=12)

    add_up_xaxis(ax)
    ax.legend(loc="upper right", fontsize=16)

    ax.grid(False)
    fig.savefig("{}.pdf".format(outname))

    # make a ratio plot
    fig, ax = get_plot()
    xvals = [0.5 * (x[1] + x[0]) for x in pairwise(bins)]
    xerrs = [0.5 * (x[1] - x[0]) for x in pairwise(bins)]

    # FIXME: Since we have 4 arrays not 3, addressed in above function
    for idx in range(1, len(arrays)):
        ratio, ratio_err = get_ratio(vals_list[-1], vals_list[idx - 1])
        label = None if ratio_legends is None else ratio_legends[idx - 1]
        ax.errorbar(
            xvals, ratio, yerr=ratio_err, fmt="o", xerr=xerrs, lw=2, label=label
        )

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ratio_label, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.tick_params(axis="both", which="minor", labelsize=12)

    ax.set_ylim([0.0, 1.2])
    add_up_xaxis(ax)

    if ratio_legends is not None:
        ax.legend(loc="lower left", fontsize=16)

    ax.grid(False)
    fig.savefig("{}_ratio.pdf".format(outname))
