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

sys.path.append("src")

inputdir = "./run_all/fwp_feature_store"
proc_files = sorted(glob.glob(os.path.join(inputdir, "*")))

i = 0
t = 0

true_e = []
false_e = []
input_e = []

print("     Total Files:", len(proc_files))
for f in proc_files:
    i = i + 1
    if i != 0 and i % 1000 == 0:
        print("Processed Events:", i)

    # hits = trackml.dataset.load_event_hits(f)
    # nhits = hits.hit_id.count()
    feature_data = torch.load(f, map_location="cpu")

    # get true_edges
    e = feature_data.edge_index
    pid = feature_data.pid
    truth = pid[e[0]] == pid[e[1]]

    true_edges = e[:, truth]
    false_edges = e[:, ~truth]

    # print("edge_index : ", e.shape[1])
    # print("true_edges : ", true_edges.shape[1])
    # print("false_edges: ", false_edges.shape[1])

    input_e.append(e.shape[1])
    true_e.append(true_edges.shape[1])
    false_e.append(false_edges.shape[1])

print("# of input_edges: ", sum(input_e))
print("# of true_edges : ", sum(true_e))
print("# of false_edges: ", sum(false_e))

colors = ["blue", "green", "orange"]
names = ["All Edges", "False Edges", "True Edges"]
x1 = np.asarray(input_e)
x2 = np.asarray(false_e)
x3 = np.asarray(true_e)

# Plot Histograms
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

nbins = 100

# All together
ax.hist(
    [x1, x2, x3],
    bins=nbins,
    color=colors,
    label=names,
    histtype="step",
    stacked=False,
    fill=False,
)

# Or, one by one
# ax.hist(x1, bins=nbins, edgecolor='None', alpha=0.8, label='Total Pairs', color='blue')
# ax.hist(x2, bins=nbins, edgecolor='None', alpha=0.8, label='False Pairs', color='orange')
# ax.hist(x3, bins=nbins, edgecolor='None', alpha=0.8, label='True Pairs',  color='green')

# plotting params
# ax.set_title('Edge Construction')
ax.set_xlabel("Number of Edges", fontsize=20)
ax.set_ylabel("Counts", fontsize=20)
ax.tick_params(axis="both", which="major", labelsize=15)
ax.tick_params(axis="both", which="minor", labelsize=15)
# ax.set_xscale('log')
ax.set_yscale("log")
# ax.set_xlim(-41, 41)
# ax.set_ylim(-41, 41)
ax.grid(False)
ax.legend(fontsize=20, loc="best")
fig.tight_layout()
fig.savefig("edges.pdf")
fig.show()
