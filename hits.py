#!/usr/bin/env python
# coding: utf-8
import sys, os, glob, yaml
import math
import random
import numpy as np
import pandas as pd
import trackml.dataset
import torch


import ROOT

# %jsroot on


inputdir = "./run_all/fwp_feature_store"
proc_files = sorted(glob.glob(os.path.join(inputdir, "*")))


h1 = ROOT.TH1F("h1", "Number of Hits;No. of Hits;Counts", 200, 100, 500)
h2 = ROOT.TH1F("h2", "Number of Hits;No. of Hits;Counts", 200, 100, 500)


i = 0

# Loop over events
print("     Total Files:", len(proc_files))
for f in proc_files[1000]:
    i = i + 1
    if i != 0 and i % 1000 == 0:
        print("Processed Events:", i)

    # load event
    feature_data = torch.load(f, map_location="cpu")
    nhits = feature_data.hid.size(0)

    # fill hist
    h1.Fill(nhits)


# PyROOT: TCanvas
can = ROOT.TCanvas("can", "", 600, 500)
can.cd()
h1.GetXaxis().SetTitleOffset(1.5)
h1.Draw()
can.Draw()
can.SaveAs("hits.pdf")
can.SaveAs("hits.C")
can.Close()
