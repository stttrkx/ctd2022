#!/usr/bin/env python
# coding: utf-8

import os
import logging
import torch
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


"""Class-based Callback Inference for Integration with Pytorch Lightning"""


# EmbeddingMetrics Callback
class EmbeddingMetrics(Callback):
    """Simpler version of 'GNNTelemetry' callback. It contains standardised
    tests (AUC-ROC & AUC-PRC curves) of the performance of a GNN network."""

    def __init__(self):
        super().__init__()
        self.preds, self.truth = None, None
        logging.info("CONSTRUCTING CALLBACK!")

    def on_test_start(self, trainer, pl_module):
        """This hook is automatically called when the model is tested
        after training. The best checkpoint is automatically loaded"""
        self.preds = []
        self.truth = []

        print("Starting GNNMetrics...")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Get the relevant outputs from each batch"""

        self.preds.append(outputs["score"].cpu())  # ADAK: preds to score
        self.truth.append(outputs["truth"].cpu())

    def on_test_end(self, trainer, pl_module):
        """
        1. Aggregate all outputs,
        2. Calculate the ROC/PRC curve,
        3. Plot ROC/PRC curve,
        4. Save plots to PDF 'AUC-ROC/PRC.pdf'
        """

        # REFACTOR THIS INTO CALCULATE METRICS, PLOT METRICS, SAVE METRICS

        # Output Directory
        output_dir = pl_module.hparams.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Aggregate 'truth' and 'pred' from all batches.
        preds = torch.cat(self.preds)
        truth = torch.cat(self.truth)
        print("preds: {}, truth: {}".format(preds.shape, truth.shape))

        # ----- ROC Metric
        # fpr, tpr, threshold = roc_curve(truth, preds)
        roc_fpr, roc_tpr, roc_thr = roc_curve(truth, preds)
        roc_auc = auc(roc_fpr, roc_tpr)
        logging.info("ROC AUC: %s", roc_auc)

        # Plotting
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axs = axs.flatten() if type(axs) is list else [axs]

        axs[0].plot(
            roc_fpr,
            roc_tpr,
            color="darkorange",
            label="ROC Curve, AUC = %.5f" % roc_auc,
        )
        axs[0].plot([0, 1], [0, 1], color="navy", linestyle="--")
        axs[0].set_xlabel("False Positive Rate", fontsize=20)
        axs[0].set_ylabel("True Positive Rate", fontsize=20)
        axs[0].set_title("ROC Curve, AUC = %.5f" % roc_auc)
        axs[0].legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "curve_roc.pdf"), format="pdf")

        # ----- PRC Metric
        # ppv, tpr, thr = precision_recall_curve(truth, preds)
        pre, recall, thr = precision_recall_curve(truth, preds)
        prc_auc = auc(recall, pre)
        logging.info("PRC AUC: %s", prc_auc)

        # Plotting
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axs = axs.flatten() if type(axs) is list else [axs]

        axs[0].plot(
            recall, pre, color="darkorange", label="PR Curve, AUC = %.5f" % prc_auc
        )
        axs[0].plot([0, 1], [1, 0], color="navy", linestyle="--")
        axs[0].set_xlabel("Recall", fontsize=20)
        axs[0].set_ylabel("Precision", fontsize=20)
        axs[0].set_title("PR Curve, AUC = %.5f" % prc_auc)
        axs[0].legend(loc="lower left")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "curve_prc.pdf"), format="pdf")

        # ----- Eff-Pur Metric
        eff = roc_tpr
        pur = 1 - roc_fpr
        epc_auc = auc(eff, pur)
        logging.info("EPC AUC: %s", epc_auc)

        # Plotting
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axs = axs.flatten() if type(axs) is list else [axs]

        axs[0].plot(
            eff, pur, color="darkorange", label="EP Curve, AUC = %.5f" % epc_auc
        )
        axs[0].plot([0, 1], [1, 0], color="navy", linestyle="--")
        axs[0].set_xlabel("Efficiency", fontsize=20)
        axs[0].set_ylabel("Purity", fontsize=20)
        axs[0].set_title("EP Curve, AUC = %.5f" % epc_auc)
        axs[0].legend(loc="lower left")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "curve_epc.pdf"), format="pdf")


class EmbeddingMetrics_v2(Callback):
    """Simpler version of 'GNNTelemetry' callback. It contains standardised
    tests (AUC-ROC & AUC-PRC curves) of the performance of a GNN network."""

    def __init__(self):
        super().__init__()
        self.preds, self.truth = None, None
        logging.info("Constructing GNNMetrics Callback !")

    def on_test_start(self, trainer, pl_module):
        """This hook is automatically called when the model is tested
        after training. The best checkpoint is automatically loaded"""
        self.preds = []
        self.truth = []

        print("Starting GNNMetrics...")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Get the relevant outputs from each batch"""

        self.preds.append(outputs["score"])  # ADAK: preds to score
        self.truth.append(outputs["truth"])

    def on_test_end(self, trainer, pl_module):
        """
        1. Aggregate all outputs,
        2. Calculate the ROC/PRC/EPC curve,
        3. Plot ROC/PRC/EPC curve,
        4. Save plots to PDF 'AUC-ROC/PRC.pdf'
        """

        # REFACTOR THIS INTO CALCULATE METRICS, PLOT METRICS, SAVE METRICS

        # Output Directory
        output_dir = pl_module.hparams.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Aggregate 'truth' and 'pred' from all batches.
        preds = torch.cat(self.preds)
        truth = torch.cat(self.truth)
        print("preds: {}, truth: {}".format(preds.shape, truth.shape))

        # ----- ROC Metric
        # fpr, tpr, threshold = roc_curve(truth, preds)
        roc_fpr, roc_tpr, roc_thr = roc_curve(truth, preds)
        roc_auc = auc(roc_fpr, roc_tpr)
        logging.info("ROC AUC: %s", roc_auc)

        fig, axs = self.make_plot(
            x_val=roc_fpr,
            y_val=roc_tpr,
            x_lab="FPR",
            y_lab="TPR",
            title="ROC Curve, AUC = %.5f" % roc_auc,
            loc="lower right",
        )
        # Plotting: ROC
        axs[0].plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "roc_curve.pdf"), format="pdf")

        # ----- PRC Metric
        # ppv, tpr, thr = precision_recall_curve(truth, preds)
        pre, recall, thr = precision_recall_curve(truth, preds)
        prc_auc = auc(recall, pre)
        logging.info("PRC AUC: %s", prc_auc)

        # Plotting PRC
        fig, axs = self.make_plot(
            x_val=recall,
            y_val=pre,
            x_lab="Recall",  # TPR
            y_lab="Precision",  # PPV
            title="PR Curve, AUC = %.5f" % prc_auc,
            loc="lower left",
        )

        axs[0].plot([0, 1], [1, 0], color="navy", linestyle="--")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "prc_curve.pdf"), format="pdf")

        # ----- Eff-Pur Metric
        eff = roc_tpr
        pur = 1 - roc_fpr
        eff_pur_auc = auc(eff, pur)
        logging.info("EPC AUC: %s", eff_pur_auc)

        # Plotting: Eff-Pur
        fig, axs = self.make_plot(
            x_val=eff,
            y_val=pur,
            x_lab="Efficiency",  # TPR
            y_lab="Purity",  # TNR = 1 - FPR
            title="EP Curve, AUC = %.5f" % eff_pur_auc,
            loc="lower left",
        )

        axs[0].plot([0, 1], [1, 0], color="navy", linestyle="--")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "epc_curve.pdf"), format="pdf")

    def make_plot(self, x_val, y_val, x_lab, y_lab, title, loc):
        """common function for creating plots"""

        # init subplots
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axs = axs.flatten() if type(axs) is list else [axs]

        # plotting: data
        axs[0].plot(x_val, y_val, color="darkorange", label=title)

        # plotting: params
        axs[0].set_xlabel(x_lab, fontsize=20)
        axs[0].set_ylabel(y_lab, fontsize=20)
        axs[0].set_xlim(-0.04, 1.04)
        axs[0].set_ylim(-0.04, 1.04)
        axs[0].set_title(title)
        axs[0].legend(loc=loc)
        return fig, axs
