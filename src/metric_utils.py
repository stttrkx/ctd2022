#!/usr/bin/env python
# coding: utf-8

"""Helper functions from exatrkx-ctd2020/.../data_utils.py repo"""

import logging
from collections import namedtuple
import matplotlib.pyplot as plt
import sklearn.metrics

# Define our Metrics class as a namedtuple
Metrics = namedtuple(
    "Metrics",
    [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "prc_precision",
        "prc_recall",
        "prc_thresh",
        "prc_auc",
        "roc_fpr",
        "roc_tpr",
        "roc_thresh",
        "roc_auc",
    ],
)


def compute_metrics(preds, targets, threshold=0.5):
    """Input preds, target as Numpy"""
    # Decision boundary metrics
    y_pred, y_true = (preds > threshold), (targets > threshold)

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    # precision = sklearn.metrics.precision_score(y_true, y_pred)
    # recall = sklearn.metrics.recall_score(y_true, y_pred)
    precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )

    # Precision-Recall Curve
    prc_precision, prc_recall, prc_thresh = sklearn.metrics.precision_recall_curve(
        y_true, preds
    )
    prc_auc = sklearn.metrics.auc(prc_recall, prc_precision)

    # ROC Curve
    roc_fpr, roc_tpr, roc_thresh = sklearn.metrics.roc_curve(y_true, preds)
    roc_auc = sklearn.metrics.auc(roc_fpr, roc_tpr)

    # Organize metrics into a namedtuple
    metrics = Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        prc_precision=prc_precision,
        prc_recall=prc_recall,
        prc_thresh=prc_thresh,
        prc_auc=prc_auc,
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        roc_thresh=roc_thresh,
        roc_auc=roc_auc,
    )
    return metrics


def plot_metrics(preds, targets, metrics, name="gnn", scale="linear"):
    # Prepare the values
    labels = targets > 0.5

    # Create the Figure
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 5))

    # Plot Model Outputs
    binning = dict(bins=25, range=(0, 1), histtype="step", log=True)
    ax0.hist(preds[labels == True], label="Real Edges", **binning)
    ax0.hist(preds[labels == False], label="Fake Edges", **binning)
    ax0.set_xlabel("Model Output", size=14)
    ax0.legend(loc="best")

    # Plot Precision & Recall
    ax1.plot(
        metrics.prc_thresh, metrics.prc_precision[:-1], label="Edge Purity"
    )  # Purity
    ax1.plot(
        metrics.prc_thresh, metrics.prc_recall[:-1], label="Edge Efficiency"
    )  # Efficiency
    ax1.set_xlabel("Edge Score Cut", size=14)  # Model Threshold
    ax1.legend(loc="best")

    # Plot the ROC curve
    ax2.plot(
        metrics.roc_fpr,
        metrics.roc_tpr,
        color="darkorange",
        label="AUC = %.5f" % metrics.roc_auc,
    )
    ax2.plot([0, 1], [0, 1], color="navy", linestyle="--")
    ax2.set_xlabel("False Positive Rate", size=14)
    ax2.set_ylabel("True Positive Rate", size=14)
    ax2.set_xscale(scale)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(name + "_metrics.pdf")


def plot_roc(metrics, name="gnn"):
    """ROC Curve: Plot of FPR (x) vs TPR (y)"""
    # Figure & Axes
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    # ROC Curve with AUC
    axs.plot(
        metrics.roc_fpr, metrics.roc_tpr, color="darkorange", label="GNN Classifier"
    )
    axs.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Naive Classifier")

    # Axes Params
    axs.set_title("ROC Curve, AUC = %.5f" % metrics.roc_auc, fontsize=15)
    axs.set_xlabel("False Positive Rate", size=20)
    axs.set_ylabel("True Positive Rate", size=20)
    axs.tick_params(axis="both", which="major", labelsize=12)
    axs.tick_params(axis="both", which="minor", labelsize=12)
    axs.legend(fontsize=14, loc="lower right")

    # Fig Params
    fig.tight_layout()
    fig.savefig(name + "_roc.pdf", format="pdf")


def plot_prc(metrics, name="gnn"):
    """PR Curve: Plot of Recall (x) vs Precision (y)."""

    # Figure & Axes
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    # ROC Curve with AUC
    axs.plot(
        metrics.prc_recall,
        metrics.prc_precision,
        color="darkorange",
        label="GNN Classifier",
    )
    axs.plot([0, 1], [0, 0], color="navy", linestyle="--", label="Naive Classifier")

    # Axes Params
    axs.set_title("PRC Curve, AUC = %.5f" % metrics.prc_auc, fontsize=15)
    axs.set_xlabel("Recall", size=20)
    axs.set_ylabel("Precision", size=20)
    axs.tick_params(axis="both", which="major", labelsize=12)
    axs.tick_params(axis="both", which="minor", labelsize=12)
    axs.legend(fontsize=14, loc="center right")

    # Fig Params
    fig.tight_layout()
    fig.savefig(name + "_prc.pdf", format="pdf")


def plot_prc_thr(metrics, name="gnn"):
    """PRC Curve: Plot of Threshold (x) vs Precision & Recall (y)"""

    # Efficiency, Purity and Threshold
    eff = metrics.prc_recall[:-1]
    pur = metrics.prc_precision[:-1]
    score_cuts = metrics.prc_thresh

    # Potting
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    axs.plot(score_cuts, pur, color="darkblue", label="Edge Purity")  # Purity
    axs.plot(score_cuts, eff, color="darkorange", label="Edge Efficiency")  # Efficiency

    # Axes Params
    axs.set_title("Edge Scores vs Efficiency and Purity", fontsize=15)
    axs.set_xlabel("Edge Score Cut", size=20)
    axs.tick_params(axis="both", which="major", labelsize=12)
    axs.tick_params(axis="both", which="minor", labelsize=12)
    # axs.set_ylim(0.5,1.02)
    axs.legend(fontsize=14, loc="lower center")

    # Fig Params
    fig.tight_layout()
    fig.savefig(name + "_prc_cut.pdf", format="pdf")


def plot_epc(metrics, name="gnn"):
    """EPC Curve: Plott Efficiency (x) vs Purity (y)"""

    # Efficiency, Purity
    eff = metrics.roc_tpr
    pur = 1 - metrics.roc_fpr
    epc_auc = sklearn.metrics.auc(eff, pur)
    logging.info("EPC AUC: %s", epc_auc)

    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    axs.plot(eff, pur, color="darkorange", label="DNN Classifier, AUC = %.5f" % epc_auc)
    axs.plot([0, 1], [1, 0], color="navy", linestyle="--", label="Naive Classifier")

    # Axes Params
    # axs.set_title("EP Curve, AUC = %.5f" % epc_auc, fontsize=15)
    axs.set_xlabel("Edge Efficiency", fontsize=24)
    axs.set_ylabel("Edge Purity", fontsize=24)
    axs.tick_params(axis="both", which="major", labelsize=14)
    axs.tick_params(axis="both", which="minor", labelsize=14)
    axs.legend(loc="lower left", fontsize=20)

    # Fig Params
    fig.tight_layout()
    fig.savefig(name + "_epc.pdf", format="pdf")


def plot_epc_cut(metrics, name="gnn"):
    """EPC Curve: Plott Edge Score (x) vs Efficiency & Purity (y)"""

    # Efficiency, Purity and Threshold
    eff = metrics.roc_tpr
    pur = 1 - metrics.roc_fpr
    score_cuts = metrics.roc_thresh

    # Make sure this is nicely plottable!
    eff, pur, score_cuts = (
        eff[score_cuts <= 1],
        pur[score_cuts <= 1],
        score_cuts[score_cuts <= 1],
    )

    # Plotting
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    axs.plot(score_cuts, pur, color="darkblue", label="Edge Purity")  # Purity
    axs.plot(score_cuts, eff, color="darkorange", label="Edge Efficiency")  # Efficiency

    # Axes Params
    # axs.set_title("Edge Scores vs Efficiency and Purity", fontsize=15)
    axs.set_xlabel("Edge Score Cut", fontsize=24)
    axs.tick_params(axis="both", which="major", labelsize=14)
    axs.tick_params(axis="both", which="minor", labelsize=14)
    # axs.set_ylim(0.5,1.02)
    axs.legend(loc="lower center", fontsize=20)

    # Fig Params
    fig.tight_layout()
    fig.savefig(name + "_epc_cut.pdf", format="pdf")


def plot_output(preds, targets, threshold=0.5, name="gnn"):
    # Prepare the values
    labels = targets > threshold

    # Figure & Axes
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    # Ploting
    binning = dict(bins=25, range=(0, 1), histtype="step", log=True)
    axs.hist(preds[labels == True], label="True Edges", **binning)  # True Edges
    axs.hist(preds[labels == False], label="False Edges", **binning)  # False Edges

    # Axes Params
    # axs.set_title("Classifier Output", fontsize=15)
    axs.set_xlabel("Model Output", size=20)
    axs.set_ylabel("Counts", size=20)
    axs.tick_params(axis="both", which="major", labelsize=14)
    axs.tick_params(axis="both", which="minor", labelsize=14)
    # axs.set_ylim(ymin=.005)
    axs.legend(loc="upper center", fontsize=16)

    # Fig Params
    fig.tight_layout()
    fig.savefig(name + "_outputs.pdf")


# Draw Prediction Sample
def draw_sample_xy(hits, edges, preds, labels, cut=0.5, figsize=(16, 16)):
    """ "Draw Sample with True and False Edges"""

    # coordinate transformation
    r, phi, ir = hits.T
    x, y = polar_to_cartesian(r, phi)

    # detector layout
    fig, ax = detector_layout(figsize=figsize)

    # Draw the segments
    for j in range(labels.shape[0]):

        ptx1 = x[edges[0, j]]
        ptx2 = x[edges[1, j]]
        pty1 = y[edges[0, j]]
        pty2 = y[edges[1, j]]

        # False Negatives
        if preds[j] < cut and labels[j] > cut:
            # ax.plot([x[edges[0,j]], x[edges[1,j]]], [y[edges[0,j]], y[edges[1,j]]], '--', c='b')
            ax.plot([ptx1, ptx2], [pty1, pty2], "--", color="b", lw=1.5, alpha=0.9)

        # False Positives
        if preds[j] > cut and labels[j] < cut:
            # ax.plot([x[edges[0,j]], x[edges[1,j]]], [y[edges[0,j]], y[edges[1,j]]], '-', c='r', alpha=preds[j])
            ax.plot([ptx1, ptx2], [pty1, pty2], "-", color="r", lw=1.5, alpha=0.15)

        # True Positives
        if preds[j] > cut and labels[j] > cut:
            # ax.plot([x[edges[0,j]], x[edges[1,j]]], [y[edges[0,j]], y[edges[1,j]]], '-', c='k', alpha=preds[j])
            ax.plot([ptx1, ptx2], [pty1, pty2], "-", color="k", lw=1.5, alpha=0.9)

    return fig, ax
