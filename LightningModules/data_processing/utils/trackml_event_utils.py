#!/usr/bin/env python
# coding: utf-8

"""The module contains useful functions for handling data at the event level.
More fine-grained utilities are reserved for detector_utils and cell_utils."""

import os
import logging
import torch
import trackml.dataset
import numpy as np
from torch_geometric.data import Data

from .event_utils import (
    get_layerwise_edges,
    get_modulewise_edges,
    get_orderwise_edges,
    get_time_ordered_true_edges,
)

from .heuristic_utils import get_layerwise_graph, get_all_edges, graph_intersection

device = "cuda" if torch.cuda.is_available() else "cpu"


def process_particles(particles):
    """Particles has duplicated records, drop duplicates."""
    particles["nhits"] = particles.groupby(["particle_id"])["nhits"].transform("count")
    particles.loc[particles.particle_id == 0, "nhits"] = -1
    particles.drop_duplicates(inplace=True, ignore_index=True)
    return particles


def process_truth(truth):
    """Create missing columns in the truth dataframe."""

    if "nhits" not in truth.columns:
        truth["nhits"] = truth.groupby("particle_id")["particle_id"].transform("count")
        truth.loc[truth.particle_id == 0, "nhits"] = -1

    if "weight" not in truth.columns:
        # Count occurrences of each hit_id
        hit_counts = truth["hit_id"].value_counts()

        # Create weight mapping for each hit_id
        hit_weights = 1 / len(hit_counts)

        # Map weights to original dataframe
        truth["weight"] = truth["hit_id"].map(lambda x: hit_weights)

    return truth


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

    # Load data using event_prefix (e.g. path/to/event0000000001)
    hits, tubes, particles, truth = trackml.dataset.load_event(event_prefix)

    logging.info("Loading event {} from CSV data source".format(event_prefix))

    # Preserve original index as pseudo time
    hits["stime"] = hits.index

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
    # FIXME: Create missing columns (nhits, weight) in `truth`
    # truth = process_truth(truth)

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

    # Get edge weight (Not used yet, hence related code is commented out)
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
        # hits.weight.to_numpy(),
        # edge_weight_norm,
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
                # hit_weights,
                # edge_weights
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
                # hit_weights=torch.from_numpy(hit_weights),
                # edge_weights=torch.from_numpy(edge_weights),
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
