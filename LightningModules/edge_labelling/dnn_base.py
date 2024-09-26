#!/usr/bin/env python
# coding: utf-8

# NOTE: dnn_base is exactly same as gnn_base (from ctd2022p repo.)

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from .utils.data_utils import split_datasets
from sklearn.metrics import roc_auc_score, accuracy_score


def roc_auc_score_robust(y_true, y_pred):
    # Handle if y_true holds only one class
    if len(np.unique(y_true)) == 1:
        return accuracy_score(y_true, np.rint(y_pred))
    else:
        return roc_auc_score(y_true, y_pred)


class DNNBase(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        """Initialise LightningModule to scan different GNN training regimes"""

        # Assign hyperparameters
        self.save_hyperparameters(hparams)

        # Set workers from hparams
        self.n_workers = (
            self.hparams["n_workers"]
            if "n_workers" in self.hparams
            else len(os.sched_getaffinity(0))
        )

        # Instance Variables
        self.trainset, self.valset, self.testset = None, None, None

    def setup(self, stage="fit"):
        if self.trainset is None:
            self.trainset, self.valset, self.testset = split_datasets(**self.hparams)

    # Data Loaders
    def train_dataloader(self):
        if self.trainset is not None:
            return DataLoader(
                self.trainset, batch_size=1, num_workers=self.n_workers
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def val_dataloader(self):
        if self.valset is not None:
            return DataLoader(
                self.valset, batch_size=1, num_workers=self.n_workers
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    def test_dataloader(self):
        if self.testset is not None:
            return DataLoader(
                self.testset, batch_size=1, num_workers=self.n_workers
            )  # , pin_memory=True, persistent_workers=True)
        else:
            return None

    # Configure Optimizer & Scheduler
    def configure_optimizers(self):
        """Configure the Optimizer and Scheduler"""
        optimizer = [
            torch.optim.AdamW(
                self.parameters(),
                lr=(self.hparams["lr"]),
                betas=(0.9, 0.999),
                eps=1e-08,
                amsgrad=True,
            )
        ]
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=self.hparams["patience"],
                    gamma=self.hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
        return optimizer, scheduler

    # 1 - Helper Function
    def get_input_data(self, batch):

        if self.hparams["cell_channels"] > 0:
            input_data = torch.cat(
                [batch.cell_data[:, : self.hparams["cell_channels"]], batch.x], dim=-1
            )
            input_data[input_data != input_data] = 0
        else:
            input_data = batch.x
            input_data[input_data != input_data] = 0

        return input_data

    # 2 - Helper Function
    def handle_directed(self, batch, edge_sample, truth_sample):

        edge_sample = torch.cat([edge_sample, edge_sample.flip(0)], dim=-1)
        truth_sample = truth_sample.repeat(2)

        if ("directed" in self.hparams.keys()) and self.hparams["directed"]:
            direction_mask = batch.x[edge_sample[0], 0] < batch.x[edge_sample[1], 0]
            edge_sample = edge_sample[:, direction_mask]
            truth_sample = truth_sample[direction_mask]

        return edge_sample, truth_sample

    # 3 - Helper Function
    def log_metrics(self, score, preds, truth, batch, loss):

        edge_positive = preds.sum().float()
        edge_true = truth.sum().float()
        edge_true_positive = (truth.bool() & preds).sum().float()

        eff = edge_true_positive.clone().detach() / max(1, edge_true)
        pur = edge_true_positive.clone().detach() / max(1, edge_positive)

        # special function to handle classes in y_true
        auc = roc_auc_score_robust(truth.bool().cpu().detach(), score.cpu().detach())

        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {
                "val_loss": loss,
                "auc": auc,
                "eff": eff,
                "pur": pur,
                "current_lr": current_lr,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=10240,
        )

    # Train Step
    def training_step(self, batch, batch_idx):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        truth = (
            batch.y_pid.bool() if "pid" in self.hparams["regime"] else batch.y.bool()
        )

        edge_sample, truth_sample = self.handle_directed(batch, batch.edge_index, truth)
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None

        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), weight=manual_weights, pos_weight=weight
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=10240,
        )

        return loss

    # Shared Evaluation for Validation and Test Steps
    def shared_evaluation(self, batch, batch_idx, log=False):

        weight = (
            torch.tensor(self.hparams["weight"])
            if ("weight" in self.hparams)
            else torch.tensor((~batch.y_pid.bool()).sum() / batch.y_pid.sum())
        )

        truth = (
            batch.y_pid.bool() if "pid" in self.hparams["regime"] else batch.y.bool()
        )

        edge_sample, truth_sample = self.handle_directed(batch, batch.edge_index, truth)
        input_data = self.get_input_data(batch)
        output = self(input_data, edge_sample).squeeze()

        if "weighting" in self.hparams["regime"]:
            manual_weights = batch.weights
        else:
            manual_weights = None

        loss = F.binary_cross_entropy_with_logits(
            output, truth_sample.float(), weight=manual_weights, pos_weight=weight
        )

        # Edge filter performance
        score = torch.sigmoid(output)
        preds = score > self.hparams["edge_cut"]

        if log:
            self.log_metrics(score, preds, truth_sample, batch, loss)

        return {"loss": loss, "score": score, "preds": preds, "truth": truth_sample}

    # Validation Step
    def validation_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=True)

        return outputs["loss"]

    # Test Step
    def test_step(self, batch, batch_idx):

        outputs = self.shared_evaluation(batch, batch_idx, log=False)

        return outputs

    # Optimizer Step
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx=0,  # ADAK: optimizer_idx to optimizer_idx=0
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (
            self.trainer.current_epoch
            < self.hparams["warmup"]  # ADAK: global_step > current_epoch
        ):
            lr_scale = min(
                1.0, float(self.trainer.current_epoch + 1) / self.hparams["warmup"]
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
