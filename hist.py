#!/usr/bin/env python
# coding: utf-8
import sys, os, glob, yaml
import math
import random
import numpy as np
import pandas as pd
import trackml.dataset
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from src import Compose_Event

sys.path.append("src")

inputdir = "./run_all/fwp_feature_store"
proc_files = sorted(glob.glob(os.path.join(inputdir, "*")))


i = 0
pt_l = []
nhits_l = []

# Loop over events
print("     Total Files:", len(proc_files))
for f in proc_files:
    i = i + 1
    if i != 0 and i % 1000 == 0:
        print("Processed Events:", i)

    # load event
    feature_data = torch.load(f, map_location="cpu")

    # get nhits and pt
    nhits_l.append(feature_data.hid.size(0))

print("# of nhits: ", sum(nhits_l))

# To numpy()
nhits = np.asarray(nhits_l)

# Plot Histograms
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

ax.hist(
    nhits,
    bins=100,
    edgecolor="blue",
    alpha=0.8,
    label="Nhits",
    color="blue",
    histtype="step",
)

# params
ax.set_xlabel("Nhits", fontsize=20)
ax.set_ylabel("Counts", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=15)
ax.tick_params(axis="both", which="minor", labelsize=15)
ax.set_yscale("log")
ax.grid(False)
ax.legend(fontsize=20, loc="best")
fig.tight_layout()
fig.savefig("nhits.pdf")
fig.show()
