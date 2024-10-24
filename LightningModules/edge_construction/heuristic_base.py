#!/usr/bin/env python
# coding: utf-8

import os
from pytorch_lightning import LightningDataModule


# TODO: Idea is to use a Heuristic Method for graph construction. Currently, this
# is done in prepare_event() in data_processing since is easy to implement there.

class HeuristicBase(LightningDataModule):

    def __init__(self, hparams):

        super().__init__()

        # Load hyperparameters
        self.save_hyperparameters(hparams)

        # Set required hyperparameters
        self.input_dir = self.hparams["input_dir"]
        self.output_dir = self.hparams["output_dir"]
        self.n_files = self.hparams["n_files"]
        self.skewed = self.hparams["skewed"]
        self.chunksize = self.hparams["chunksize"]
        self.n_tasks = self.hparams["n_tasks"]

        # Set defaults for optional hyperparameters
        self.task = self.hparams["task"] if "task" in self.hparams else 0
        self.n_workers = (
            self.hparams["n_workers"]
            if "n_workers" in self.hparams
            else len(os.sched_getaffinity(0))
        )
        self.build_weights = (
            self.hparams["build_weights"] if "build_weights" in self.hparams else True
        )
        self.show_progress = (
            self.hparams["show_progress"] if "show_progress" in self.hparams else True
        )
