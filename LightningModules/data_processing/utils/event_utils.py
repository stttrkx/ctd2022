#!/usr/bin/env python
# coding: utf-8

"""The module contains useful functions for handling data at the event level.
More fine-grained utilities are reserved for detector_utils and cell_utils."""

import os
import logging
import itertools
import torch
import trackml.dataset
import numpy as np
from torch_geometric.data import Data

from .heuristic_utils import get_layerwise_graph, get_all_edges, graph_intersection

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_time_ordered_true_edges(hits):
    """Get time ordered true edge list.  Here 'hits' represent complete event."""

    true_edges_start = []
    true_edges_end = []

    # Sort by the time of the MC Points
    for particle_id in hits.particle_id.unique():
        sortedHits = hits.query(f"particle_id=={particle_id}").sort_values(
            "tT", ascending=True
        )
        for hit in range(sortedHits.x.size - 1):
            true_edges_start.append(sortedHits.index[hit])
            true_edges_end.append(sortedHits.index[hit + 1])

    true_edges = np.array([true_edges_start, true_edges_end])

    # Add the reverse edges
    true_edges = np.concatenate((true_edges, true_edges[[1, 0]]), axis=1)

    return true_edges


def get_layerwise_edges(hits):
    """Get layerwise true edge list. Here 'hits' represent complete event."""

    # Sort by increasing distance from production (IP)
    hits = hits.assign(
        R=np.sqrt(
            (hits.x - hits.vx) ** 2 + (hits.y - hits.vy) ** 2 + (hits.z - hits.vz) ** 2
        )
    )
    hits = hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)

    # Handle noise (particle_id == 0)
    # hits.loc[hits["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particles and layers
    hit_list = (
        hits.groupby(["particle_id", "layer_id"], sort=False)["index"]
        .agg(lambda x: list(x))
        .groupby(level=0)
        .agg(lambda x: list(x))
    )

    # Build true edges
    true_edges = []
    for row in hit_list.values:
        for i, j in zip(row[0:-1], row[1:]):
            true_edges.extend(list(itertools.product(i, j)))

    # Convert and Transform
    true_edges = np.array(true_edges).T
    return true_edges, hits


def get_modulewise_edges(hits):
    """Get modulewise (layerless) true edge list using the order
    of hits based on R. Here 'hits' represent complete event."""

    # Handle NaN and Null Values
    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    signal = signal.drop_duplicates(
        subset=["particle_id", "volume_id", "layer_id", "module_id"]
    )

    # Sort by increasing distance from production (IP)
    signal = signal.assign(
        R=np.sqrt(
            (signal.x - signal.vx) ** 2
            + (signal.y - signal.vy) ** 2
            + (signal.z - signal.vz) ** 2
        )
    )
    signal = signal.sort_values("R").reset_index(drop=False)

    # Handle re-indexing
    signal = signal.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)

    # Handle noise (particle_id == 0)
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particles
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )

    # Build true edges
    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])

    # Convert and Transform
    true_edges = np.array(true_edges).T
    true_edges = signal.unsorted_index.values[true_edges]
    return true_edges


def get_orderwise_edges(hits):
    """Get modulewise (layerless) true edge list using the order
    of occurrence of hits. Here 'hits' represent complete event."""

    # Restore the original order using pseudo time
    hits = hits.sort_values(by="stime").reset_index(drop=True)
    hits = hits.drop(columns=["stime"])

    # Handle NaN and Null Values
    signal = hits[
        ((~hits.particle_id.isna()) & (hits.particle_id != 0)) & (~hits.vx.isna())
    ]
    signal = signal.drop_duplicates(
        subset=["particle_id", "volume_id", "layer_id", "module_id"]
    )

    # Reset index to get 'index' column
    signal = signal.reset_index(drop=False)

    # Handle re-indexing
    # signal = signal.rename(columns={"index": "unsorted_index"}).reset_index(drop=False)

    # Handle Particle_id 0
    signal.loc[signal["particle_id"] == 0, "particle_id"] = np.nan

    # Group by particles
    signal_list = signal.groupby(["particle_id"], sort=False)["index"].agg(
        lambda x: list(x)
    )

    # Build true edges
    true_edges = []
    for row in signal_list.values:
        for i, j in zip(row[:-1], row[1:]):
            true_edges.append([i, j])

    # Convert and Transform
    true_edges = np.array(true_edges).T

    # Restore the Order
    # true_edges = signal.unsorted_index.values[true_edges]

    return true_edges


# --------------------- Move to trackml_event_utils.py
def signal_selection(particles):
    """Special manipulation on particles data frame. One can exclude a particular
    particle type to treat it as a noise. For example, if 'signal' is protons & pions
    then every other particle will be treated as a noise, pass 'signal' from config."""

    # select signal
    signal = [-2212, 2212, -211, 211]  # in future pass signal from config file
    particles = particles[particles["pdgcode"].isin(signal)].reset_index(drop=True)
    return particles


def select_hits(event_prefix: str, noise: bool, skewed: bool, **kwargs):
    """Hit selection method from Exa.TrkX. Build a full event, select hits based on certain criteria."""

    # load data using event_prefix (e.g. path/to/event0000000001)
    hits, tubes, particles, truth = trackml.dataset.load_event(event_prefix)

    logging.info("Loading event {} from CSV data source".format(event_prefix))

    # Store original order (needed for orderwise_true_edges() function)
    hits["original_order"] = hits.index

    # Preprocess 'particles' to get nhits, and drop duplicates
    particles["nhits"] = particles.groupby(["particle_id"])["nhits"].transform("count")
    particles.drop_duplicates(inplace=True, ignore_index=True)

    # Apply particle selection (not nice, but works)
    if kwargs["selection"]:
        particles = signal_selection(particles)

    # Handle noise
    if noise:
        # runs if noise=True
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "pdgcode"]],
            on="particle_id",
            how="left",
        )
    else:
        # runs if noise=False
        truth = truth.merge(
            particles[["particle_id", "vx", "vy", "vz", "pdgcode"]],
            on="particle_id",
            how="inner",
        )

    # Derive new quantities from 'truth'
    px = truth.tpx
    py = truth.tpy
    pz = truth.tpz

    pt = np.sqrt(px**2 + py**2)
    ptheta = np.arctan2(pt, pz)
    peta = -np.log(np.tan(0.5 * ptheta))
    pphi = np.arctan2(py, px)

    # Add quantities as columns to 'truth'
    truth = truth.assign(pt=pt, ptheta=ptheta, peta=peta, pphi=pphi)

    # Merge 'hits' and 'tubes' DataFrames on 'hit_id'
    hits = hits.merge(
        tubes[["hit_id", "isochrone", "skewed", "sector_id"]], on="hit_id"
    )

    # Handle skewed layers
    if skewed is False:
        hits = hits.query("skewed==0")

    # Calculate derived variables from 'hits'
    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    r3 = np.sqrt(hits.x**2 + hits.y**2 + hits.z**2)
    theta = np.arccos(hits.z / r3)
    eta = -np.log(np.tan(theta / 2.0))

    # Add r, phi, theta, eta to 'hits' and merge with 'truth'
    hits = hits.assign(r=r, phi=phi, theta=theta, eta=eta).merge(truth, on="hit_id")

    # Add 'event_id' column to this event.
    hits = hits.assign(event_id=int(event_prefix[-10:]))
    return hits


def build_event(
    event_prefix,
    feature_scale,
    layerwise,
    modulewise,
    orderwise,
    timeordered,
    inputedges,
    noise,
    skewed,
    **kwargs,
):
    """Builds the event data by loading the event file and preprocessing the hit's data."""

    # Load event using "event_file" prefix (load_event function transferred to select_hits function).
    # hits, tubes, particles, truth = trackml.dataset.load_event(event_file)

    # Select hits, add new/select columns, add event_id
    hits = select_hits(event_prefix, noise=noise, skewed=skewed, **kwargs)

    # Get list of all layers
    layers = hits.layer_id.to_numpy()

    # Handle which truth graph(s) are being produced
    modulewise_true_edges, layerwise_true_edges = None, None
    orderwise_true_edges, time_ordered_true_edges = None, None

    # Get true edge list using the ordering of layers
    if layerwise:
        layerwise_true_edges, hits = get_layerwise_edges(hits)
        logging.info(
            "Layerwise truth graph built for {} with size {}".format(
                event_prefix, layerwise_true_edges.shape
            )
        )

    # Get true edge list without layer ordering
    if modulewise:
        modulewise_true_edges = get_modulewise_edges(hits)
        logging.info(
            "Modulewise truth graph built for {} with size {}".format(
                event_prefix, modulewise_true_edges.shape
            )
        )

    # Get true edge list without layer ordering (natural order)
    if orderwise:
        orderwise_true_edges = get_orderwise_edges(hits)
        logging.info(
            "Orderwise truth graph built for {} with size {}".format(
                event_prefix, orderwise_true_edges.shape
            )
        )

    # Get true edge list without layer ordering (time ordered)
    if timeordered:
        time_ordered_true_edges = get_time_ordered_true_edges(hits)
        logging.info(
            "Time ordered truth graph built for {} with size {}".format(
                event_prefix, time_ordered_true_edges.shape
            )
        )

    # Handle whether input graph(s) are being produced
    input_edges = None

    # Get input edge list using order of layers.
    if inputedges == "oldLayerwise":
        input_edges = get_layerwise_graph(
            hits, filtering=kwargs["filtering"], inneredges=False
        )  # w/o samelayer edges
        logging.info(
            "Layerwise input graph built for {} with size {}".format(
                event_prefix, input_edges.shape
            )
        )

    elif inputedges == "newLayerwise":
        input_edges = get_layerwise_graph(
            hits, filtering=kwargs["filtering"], inneredges=True
        )  # with samelayer edges
        logging.info(
            "Layerwise input graph built for {} with size {}".format(
                event_prefix, input_edges.shape
            )
        )

    elif inputedges == "all":
        input_edges = get_all_edges(hits)
        logging.info(
            "All input graph built for {} with size {}".format(
                event_prefix, input_edges.shape
            )
        )

    else:
        logging.error(f"{inputedges} is not a valid method to build input graphs")
        exit(1)

    # No weights of tracks in STT data yet, skipping it.
    # Get edge weight
    # edge_weights = (
    #    hits.weight.to_numpy()[modulewise_true_edges]
    #    if modulewise
    #    else hits.weight.to_numpy()[layerwise_true_edges]
    # )
    # edge_weight_average = (edge_weights[0] + edge_weights[1]) / 2
    # edge_weight_norm = edge_weight_average / edge_weight_average.mean()

    logging.info("Weights are not constructed, no weights for STT")

    return (
        hits[["r", "phi", "isochrone"]].to_numpy() / feature_scale,
        hits.particle_id.to_numpy(),
        layers,
        layerwise_true_edges,
        modulewise_true_edges,
        orderwise_true_edges,
        time_ordered_true_edges,
        input_edges,
        hits["hit_id"].to_numpy(),
        hits.pt.to_numpy(),
        hits[["vx", "vy", "vz"]].to_numpy(),
        hits.pdgcode.to_numpy(),
        hits.ptheta.to_numpy(),
        hits.peta.to_numpy(),
        hits.pphi.to_numpy(),
    )


def prepare_event(
    event_prefix,
    output_dir,
    layerwise,
    modulewise,
    orderwise,
    timeordered,
    inputedges,
    noise,
    skewed,
    overwrite,
    **kwargs,
):
    """Main function for processing an event in trackml format"""

    try:
        evtid = int(event_prefix[-10:])
        filename = os.path.join(output_dir, str(evtid))

        if not os.path.exists(filename) or overwrite:
            logging.info("Preparing event {}".format(evtid))

            # feature scale for X=[r,phi,z]
            feature_scale = [100, np.pi, 100]

            # build event
            (
                X,
                pid,
                layers,
                layerwise_true_edges,
                modulewise_true_edges,
                orderwise_true_edges,
                time_ordered_true_edges,
                input_edges,
                hid,
                pt,
                vertex,
                pdgcode,
                ptheta,
                peta,
                pphi,
            ) = build_event(
                event_prefix=event_prefix,
                feature_scale=feature_scale,
                layerwise=layerwise,
                modulewise=modulewise,
                orderwise=orderwise,
                timeordered=timeordered,
                inputedges=inputedges,
                noise=noise,
                skewed=skewed,
                **kwargs,
            )

            # build PyTorch Geometric (PyG) 'Data' object
            data = Data(
                x=torch.from_numpy(X).float(),
                pid=torch.from_numpy(pid),
                layers=torch.from_numpy(layers),
                hid=torch.from_numpy(hid),
                pt=torch.from_numpy(pt),
                vertex=torch.from_numpy(vertex),
                pdgcode=torch.from_numpy(pdgcode),
                ptheta=torch.from_numpy(ptheta),
                peta=torch.from_numpy(peta),
                pphi=torch.from_numpy(pphi),
                event_file=event_prefix,
            )

            # add true edges to PyTorch Geometric (PyG) 'Data' object
            if layerwise_true_edges is not None:
                data.layerwise_true_edges = torch.from_numpy(layerwise_true_edges)

            if modulewise_true_edges is not None:
                data.modulewise_true_edges = torch.from_numpy(modulewise_true_edges)

            if orderwise_true_edges is not None:
                data.orderwise_true_edges = torch.from_numpy(orderwise_true_edges)

            if time_ordered_true_edges is not None:
                data.time_ordered_true_edges = torch.from_numpy(time_ordered_true_edges)

            # NOTE: I am jumping from Processing to GNN stage, so I need ground truth (GT) of input
            # edges (edge_index). After embedding, one gets GT as y, and after filtering one gets
            # the GT in the form of 'y_pid'. As I intend to skip both the Embedding & the Filtering
            # stages, the input graph and its GT is build in Processing stage. The GNN can run after
            # either embedding or filtering stages, so it look for either 'y' or 'y_pid', existence of
            # one of these means the execution of these stages i.e. if y_pid exists in data that means
            # both embedding and filtering stages has been executed. If only 'y' exists then only
            # embedding stage has been executed. In principle, I should've only one of these in 'Data'.
            #
            # Now, for my case, I will build input graph during Processing and also add its GT to the
            # data. If the 'edge_index' is build in Processing then ground truth (y or y_pid) should
            # also be built here. The dimension of y (n) and y_pid (m) are given below, here m < n.
            #
            # y (n): after embedding along with e_radius (2,n), y.shape==e_radius.shape[1]
            # y_pid (m): after filtering along with e_radius (2,m), y_pid.shape==e_radius.shape[1]

            # add input edges to PyTorch Geometric (PyG) 'Data' object
            if input_edges is not None:
                # select true edges
                if layerwise:
                    true_edges = data.layerwise_true_edges
                elif modulewise:
                    true_edges = data.modulewise_true_edges
                elif orderwise:
                    true_edges = data.orderwise_true_edges
                elif timeordered:
                    true_edges = data.time_ordered_true_edges
                else:
                    true_edges = None

                assert true_edges is not None

                # get input graph
                input_edges = torch.from_numpy(input_edges)
                new_input_edges, y = graph_intersection(input_edges, true_edges)
                data.edge_index = new_input_edges
                # data.y = y     # if regime: [] will point to embedding
                data.y_pid = y  # if regime: [[pid]] points to filtering

            # add cell/tube information to Data, Check for STT
            # logging.info("Getting cell info")

            # if cell_information:
            #    data = get_cell_information(
            #        data, cell_features, detector_orig, detector_proc, end caps, noise
            #    )

            with open(filename, "wb") as pickle_file:
                torch.save(data, pickle_file)

        else:
            logging.info("{} already exists".format(evtid))

    except Exception as inst:
        logging.error("File:", event_prefix, "had exception", inst)
