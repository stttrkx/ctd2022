#!/usr/bin/env python
# coding: utf-8

"""STT Dataset (equivalent to TrackML Dataset)"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackml.dataset
from .detector import detector_layout

try:
    output_base = os.path.dirname(os.path.abspath(__file__))
except KeyError as e:
    print("Require the directory for outputs")
    print("Given by environment variable: TRKXOUTPUTDIR")
    raise e


# detector file
detector_path = os.path.join(output_base, "stt.csv")


# TODO: Adapt Event class to STT Dataset (from xju2-gnn/heptrkx/dataset/event.py)


# Event Class
class Event(object):
    """An object saving Event info, including hits, particles, truth and cell info"""

    def __init__(self, input_dir: str, noise: bool, skewed: bool):
        """Initialize Instance Variables in Constructor"""
        self._path = input_dir
        self._noise = noise
        self._skewed = skewed
        self._detector = None

        self._evtid = None
        self._hits = None
        self._particles = None
        self._truth = None
        self._cells = None
        self._event = None

    def read(self, evtid: int = None):
        """Read a Single Event Using an Event ID."""

        prefix = "event{:010d}".format(evtid)
        event_prefix = os.path.join(os.path.expandvars(self._path), prefix)

        try:
            all_data = trackml.dataset.load_event(
                event_prefix, parts=["hits", "particles", "truth", "cells"]
            )
        except Exception as e:
            return e

        if all_data is None:
            return False

        hits, particles, truth, cells = all_data
        hits = hits.assign(event_id=evtid)

        # add pT to particles
        px = particles.px
        py = particles.py
        pz = particles.pz
        pt = np.sqrt(px**2 + py**2)
        momentum = np.sqrt(px**2 + py**2 + pz**2)
        ptheta = np.arccos(pz / momentum)
        peta = -np.log(np.tan(0.5 * ptheta))
        particles = particles.assign(pt=pt, peta=peta)

        # assign vars
        self._evtid = evtid
        self._hits = hits
        self._particles = particles
        self._truth = truth
        self._cells = cells

        # compose event
        self.merge_truth_info_to_hits()
        return True

    @property
    def particles(self):
        return self._particles

    @property
    def hits(self):
        return self._hits

    @property
    def cells(self):
        return self._cells

    @property
    def truth(self):
        return self._truth

    @property
    def evtid(self):
        return self._evtid

    def merge_truth_info_to_hits(self):
        """Merge truth information ('truth', 'particles') to 'hits'.
        Then calculate and add derived variables to the event."""

        hits = self._hits

        # account for noise
        if self._noise:
            # runs if noise=True
            truth = self._truth.merge(self._particles, on="particle_id", how="left")
        else:
            # runs if noise=False
            truth = self._truth.merge(self._particles, on="particle_id", how="inner")

        # skip skewed tubes
        if self._skewed is False:
            hits = hits.query("skewed==0")

            # rename layers from 0,1,2...,17 & assign to "layer" column
            vlids = hits.layer_id.unique()
            n_det_layers = hits.layer_id.unique().shape[0]
            vlid_groups = hits.groupby(["layer_id"])
            hits = pd.concat(
                [
                    vlid_groups.get_group(vlids[i]).assign(layer=i)
                    for i in range(n_det_layers)
                ]
            )
            self._hits = hits.reset_index(drop=True)

        # merge 'hits' with 'truth'
        hits = hits.merge(truth, on="hit_id", how="left")

        # add new features to 'hits'
        x = hits.x
        y = hits.y
        z = hits.z
        absz = np.abs(z)
        r = np.sqrt(x**2 + y**2)  # in 2D
        r3 = np.sqrt(r**2 + z**2)  # in 3D
        phi = np.arctan2(hits.y, hits.x)
        theta = np.arccos(z / r3)
        eta = -np.log(np.tan(theta / 2.0))

        tpx = hits.tpx
        tpy = hits.tpy
        tpt = np.sqrt(tpx**2 + tpy**2)

        # add derived quantities to 'hits'
        hits = hits.assign(r=r, phi=phi, eta=eta, r3=r3, absZ=absz, tpt=tpt)
        self._event = hits

    def reconstructable_pids(self, min_hits=4):
        """Find reconstructable particles with min_hits > 4"""
        truth_particles = self.particles.merge(self.truth, on="particle_id", how="left")
        reconstructable_particles = truth_particles[truth_particles.nhits > min_hits]
        return np.unique(reconstructable_particles.particle_id)

    def filter_hits(self, layers, inplace=True):
        """Keep hits that are in the layers"""
        barrel_hits = self._hits[self._hits.layer.isin(layers)]
        if inplace:
            self._hits = barrel_hits
        return barrel_hits

    def remove_noise_hits(self, inplace=True):
        """Remove noise"""
        no_noise = self._hits[self._hits.particle_id > 0]
        if inplace:
            self._hits = no_noise
        return no_noise

    def remove_duplicated_hits(self, inplace=True):
        """Remove duplicate hits"""
        hits = self._hits.loc[
            self._hits.groupby(["particle_id", "layer"], as_index=False).r.idxmin()
        ]
        if inplace:
            self._hits = hits
            return self._hits
        else:
            return hits

    def select_hits(self, no_noise, eta_cut):
        """Select hits after applying an eta cut"""
        if no_noise:
            self.remove_noise_hits()

        self._hits = self._hits[np.abs(self._hits.eta) < eta_cut]
        return self._hits

    def count_duplicated_hits(self):
        """Count duplicate hits"""
        # sel is the number of "extra" hits
        # if not duplication, sel = 0; otherwise it is the number of duplication
        sel = (
            self._hits.groupby("particle_id")["layer"]
            .apply(lambda x: len(x) - np.unique(x).shape[0])
            .values
        )
        return sel


# Compose DataFrames into Single Event DataFrame.
def Build_Event(inputdir="", evtid=0, noise=False, skewed=True, selection=False):
    """Merge truth information ('truth', 'particles') to 'hits'.
    Then calculate and add derived variables to the event. Keep
    the necessary columns in the final dataframe."""

    # get event prefix
    prefix = "event{:010d}".format(evtid)
    event_prefix = os.path.join(os.path.expandvars(inputdir), prefix)

    # load data using event_prefix (e.g. path/to/event0000000001)
    hits, tubes, particles, truth = trackml.dataset.load_event(event_prefix)

    # drop duplicates (present due to "PndMLTracker")
    particles["nhits"] = particles.groupby(["particle_id"])["nhits"].transform("count")
    particles.drop_duplicates(inplace=True, ignore_index=True)

    # particle selection
    if selection:
        particles = particles[
            particles["pdgcode"].isin([-2212, 2212, -211, 211])
        ].reset_index(drop=True)

    # handle noise
    if noise:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "q", "nhits"]],
            on="particle_id",
            how="left",
        )
    else:
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "q", "nhits"]],
            on="particle_id",
            how="inner",
        )

    # calculate derived variables from 'truth'
    px = truth.tpx
    py = truth.tpy
    pz = truth.tpz

    pt = np.sqrt(px**2 + py**2)
    ptheta = np.arctan2(pt, pz)
    peta = -np.log(np.tan(0.5 * ptheta))

    truth = truth.assign(pt=pt, ptheta=ptheta, peta=peta)

    # merge some columns of tubes to the hits, I need isochrone, skewed & sector_id
    hits = hits.merge(
        tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id"
    )

    # handle skewed layers
    if skewed is False:
        hits = hits.query("skewed==0")

        # rename layer_ids from 0,1,2...,n
        vlids = hits.layer_id.unique()
        n_det_layers = len(vlids)
        vlid_groups = hits.groupby(["layer_id"])
        hits = pd.concat(
            [
                vlid_groups.get_group(vlids[i]).assign(layer_id=i)
                for i in range(n_det_layers)
            ]
        )

    # reset index, maybe its jumbled up
    hits = hits.reset_index(drop=True)

    # calculate derived variables from 'hits'
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    r3 = np.sqrt(hits.x**2 + hits.y**2 + hits.z**2)
    theta = np.arccos(hits.z / r3)
    eta = -np.log(np.tan(theta / 2.0))

    # merge 'hits' with 'truth', but first add r, phi, theta, eta
    hits = hits.assign(r=r, phi=phi, theta=theta, eta=eta).merge(truth, on="hit_id")

    # add 'event_id' column to this event.
    hits = hits.assign(event_id=int(event_prefix[-10:]))

    return hits


# Visualize Event (Using Object Oriented API)
def Build_Event_Viz(event=None, figsize=(10, 10), fig_type="pdf", save_fig=False):
    """Visualize a single event from Build_Event()."""

    # draw detector layout
    fig, ax = detector_layout(figsize=figsize)

    # get colormap for consistent track colors
    cmap = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "magenta",
        "black",
        "gray",
        "lime",
        "teal",
        "navy",
        "maroon",
        "olive",
        "indigo",
        "cyan",
    ]

    # get event_id and particle_id
    evtid = np.unique(event.event_id.values)
    unique_pids = np.unique(event.particle_id.values).astype(int)

    # draw particles
    for pid in unique_pids:
        df = event.loc[event.particle_id == pid]
        ax.scatter(
            df.x.values, df.y.values, color=cmap[pid], label=f"particle_id: {pid}"
        )

    # axis params
    evtid = np.unique(event.event_id.values)
    ax.set_title("Event # {}".format(evtid))
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()

    # save figure
    if save_fig:
        fig.savefig("event_{}.{}".format(evtid, fig_type))
