#!/usr/bin/env python
# coding: utf-8

# From gnn4itk repo
"""Modified version of 'tracks_from_gnn.py' (runs after 'eval_gnn_tf.py') script from
gnn4itk repo. The code breakdown of the script is given in 'stt4_seg.ipynb' notebook."""

import os
import torch
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.cluster import DBSCAN

device = "cuda" if torch.cuda.is_available() else "cpu"


# Sparse Matrix
def GetCOO_Matrix(senders, receivers, scores, num_nodes):
    """Prepare Sparse Matrix in COO format for DBSCAN"""

    # adjancy matrix with its value being the edge socre.
    e_csr = sps.csr_matrix(
        (scores, (senders, receivers)), shape=(num_nodes, num_nodes), dtype=np.float32
    )

    # rescale the duplicated edges
    e_csr.data[e_csr.data > 1] = e_csr.data[e_csr.data > 1] / 2.0

    # invert to treat score as an inverse distance
    e_csr.data = 1 - e_csr.data

    # make it symmetric
    e_csr_bi = sps.coo_matrix(
        (
            np.hstack([e_csr.tocoo().data, e_csr.tocoo().data]),
            np.hstack(
                [
                    np.vstack([e_csr.tocoo().row, e_csr.tocoo().col]),
                    np.vstack([e_csr.tocoo().col, e_csr.tocoo().row]),
                ]
            ),
        )
    )
    return e_csr_bi


# DBSCAN Clustering
def DBSCAN_Clustering(e_csr_bi, epsilon=0.25, min_samples=2):
    """ "Track Candidates using DBSCAN Clustering"""

    # DBSCAN Clustering on Ajacency Matrix
    clustering = DBSCAN(
        eps=epsilon, metric="precomputed", min_samples=min_samples
    ).fit_predict(e_csr_bi)

    # Get Labels
    labels = clustering[np.unique(e_csr_bi.tocoo().row)]
    return labels


# Main Function For DBSCAN Labelling
def dbscan_labelling(input_file, output_dir, edge_cut, **kwargs):
    """prepare a multiprocessing function for track building"""

    try:
        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])
        if not os.path.exists(output_file) or kwargs["overwrite"]:

            logging.info("Preparing event {}".format(output_file))

            # load weighted graph
            graph = torch.load(input_file, map_location=device)

            # get necessary data
            hit_id = graph.hid
            senders = graph.edge_index[0]
            receivers = graph.edge_index[1]
            scores = graph.scores

            # make lenghth of scores equal to edge_index
            scores = scores[: graph.edge_index.shape[1]]

            # number of nodes
            # num_nodes = graph.x.size(0)
            num_nodes = hit_id.shape[0]

            # apply edge score cut
            e_mask = scores > edge_cut
            scores, senders, receivers = (
                scores[e_mask],
                senders[e_mask],
                receivers[e_mask],
            )

            # prepare sparse matrix
            e_csr_bi = GetCOO_Matrix(senders, receivers, scores, num_nodes)

            # DBSCAN Clustering on Ajacency Matrix
            clustering = DBSCAN(
                eps=kwargs["epsilon"],
                metric="precomputed",
                min_samples=kwargs["min_samples"],
            ).fit_predict(e_csr_bi)

            # Get Labels
            graph.labels = clustering[np.unique(e_csr_bi.tocoo().row)]

            # create DataFrame with {'hit_id', 'track_id'} columns
            track_labels = np.vstack([np.unique(e_csr_bi.tocoo().row), graph.labels])

            track_labels = pd.DataFrame(track_labels.T)
            track_labels.columns = ["hit_id", "track_id"]

            new_hit_id = np.apply_along_axis(
                lambda x: hit_id[x], 0, track_labels.hit_id.values
            )

            predicted_tracks = pd.DataFrame.from_dict(
                {"hit_id": new_hit_id, "track_id": track_labels.track_id}
            )

            # all columns with sampe dtype
            predicted_tracks = predicted_tracks.astype(np.int32)

            # save DataFrame (predicted tracks)
            with open(output_file, "wb") as pickle_file:
                torch.save(predicted_tracks, pickle_file)

            # save predicted_tracks as torch tensor in the graph
            # labelled_graph.reco_tracks = torch.Tensor(predicted_tracks.values)

            # Save labelled graph
            # with open(output_file, "wb") as pickle_file:
            #    torch.save(labelled_graph, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)
