#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from functools import partial
from tqdm.contrib.concurrent import process_map

from ..segment_base import SegmentBase
from ..utils.ccl import ccl_labelling
from ..utils.dbscan import dbscan_labelling
from ..utils.wrangler import wrangler_labelling


# Segmentation data module specific to the TrackML pipeline
class TrackMLSegment(SegmentBase):
    def __init__(self, hparams):
        super().__init__(hparams)

    def prepare_data(self):

        all_files = [
            os.path.join(self.hparams["input_dir"], file)
            for file in os.listdir(self.hparams["input_dir"])
        ][: self.n_files]
        all_files = np.array_split(all_files, self.n_tasks)[self.task]

        os.makedirs(self.output_dir, exist_ok=True)
        print("Writing outputs to " + self.output_dir)

        # ADAK: Select a Labelling Method
        if self.method == "ccl":
            label_graph = ccl_labelling
        elif self.method == "dbscan":
            label_graph = dbscan_labelling
        else:
            label_graph = wrangler_labelling

        print("Labelling method is " + label_graph.__name__)

        process_func = partial(label_graph, **self.hparams)
        process_map(process_func, all_files, max_workers=self.n_workers)
