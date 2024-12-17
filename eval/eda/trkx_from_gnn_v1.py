#!/usr/bin/env python
# coding: utf-8

# From gnn4itk repo
"""Modified version of 'tracks_from_gnn.py' (runs after 'eval_gnn_tf.py') script from
gnn4itk repo. The code breakdown of the script is given in 'stt4_seg.ipynb' notebook."""

import os
import glob
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sps

from multiprocessing import Pool
from functools import partial
from sklearn.cluster import DBSCAN

device = "cuda" if torch.cuda.is_available() else "cpu"


def tracks_from_gnn(
    hit_ids,
    score,
    senders,
    receivers,
    edge_score_cut=0.0,
    epsilon=0.25,
    min_samples=2,
    **kwargs
):
    """Use DBSCAN to build tracks after GNN stage. Important variables are
    edge pair (sender, receiver) and edge score plus other variables."""

    n_nodes = hit_ids.shape[0]
    if edge_score_cut > 0:
        cuts = score > edge_score_cut
        score, senders, receivers = score[cuts], senders[cuts], receivers[cuts]

    # prepare the DBSCAN input, which the adjancy matrix with its value being the edge socre.
    e_csr = sps.csr_matrix(
        (score, (senders, receivers)), shape=(n_nodes, n_nodes), dtype=np.float32
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

    # DBSCAN get track candidates
    clustering = DBSCAN(
        eps=epsilon, metric="precomputed", min_samples=min_samples
    ).fit_predict(e_csr_bi)

    track_labels = np.vstack(
        [np.unique(e_csr_bi.tocoo().row), clustering[np.unique(e_csr_bi.tocoo().row)]]
    )

    track_labels = pd.DataFrame(track_labels.T)
    track_labels.columns = ["hit_id", "track_id"]

    new_hit_id = np.apply_along_axis(
        lambda x: hit_ids[x], 0, track_labels.hit_id.values
    )

    tracks = pd.DataFrame.from_dict(
        {"hit_id": new_hit_id, "track_id": track_labels.track_id}
    )
    return tracks


def process(filename, output_dir, **kwargs):
    """prepare a multiprocessing function for track building"""

    # print("Args: {}, {}, {}, {}".format(filename, output_dir, score_name, kwargs))

    # get the event_id from the filename
    # evtid = int(os.path.basename(filename))  # [:-4] was to skip .npz extension, skipped in my case.
    evtid = int(os.path.basename(filename))

    # gnn_prcessed data by GNNBuilder Callback
    graph = torch.load(filename, map_location=device)

    hit_ids = graph.hid
    senders = graph.edge_index[0]
    receivers = graph.edge_index[1]
    scores = graph.scores

    # score has twice the size of edge_index (flip(0) was used), cut-off the size
    scores = scores[: graph.edge_index.shape[1]]

    # predicted tracks from the GNN stage
    predicted_tracks = tracks_from_gnn(hit_ids, scores, senders, receivers, **kwargs)

    # all columns with sampe dtype
    predicted_tracks = predicted_tracks.astype(np.int32)

    # save reconstructed tracks into files

    # (i) DataFrame to Torch
    output_file = os.path.join(output_dir, "{}".format(evtid))
    with open(output_file, "wb") as pickle_file:
        torch.save(predicted_tracks, pickle_file)

    # (ii) DataFrame to CSV
    # output_file = os.path.join(output_dir, "{}.csv".format(evtid))
    # predicted_tracks.to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Tracks after GNN Evaluation (GNN Test Step)."
    )
    add_arg = parser.add_argument

    # Bookkeeping
    add_arg(
        "--input-dir",
        help="Directory saving results from evaluating GNNs",
        required=True,
    )

    # add_arg("--torch-data-dir", help='torch data directory', required=True)
    add_arg(
        "--output-dir",
        help="Output file directory for track candidates.",
        required=True,
    )
    add_arg(
        "--max-evts",
        help="Maximum number of events for track building.",
        type=int,
        default=1,
    )
    add_arg("--num-workers", help="Number of workers/threads.", default=8, type=int)
    add_arg(
        "--score-name",
        help="Edge score, either GNN score or truth",
        default="score",
        choices=["score", "truth"],
    )

    # Hyperparameters for DBScan
    add_arg("--edge-score-cut", help="edge score cuts", default=0, type=float)
    add_arg("--epsilon", help="epsilon in DBScan", default=0.25, type=float)
    add_arg(
        "--min-samples", help="minimum number of samples in DBScan", default=2, type=int
    )

    # Collect Arguments
    args = parser.parse_args()

    # Input/Output Data
    inputdir = args.input_dir  # gnn_processed/test or gnn_processed/pred
    outputdir = args.output_dir  # gnn_segmenting or seg_processed
    os.makedirs(outputdir, exist_ok=True)  # create outputdir if it doesn't exist

    all_files = glob.glob(os.path.join(inputdir, "*"))
    all_files = sorted(all_files)

    n_tot_files = len(all_files)
    max_evts = args.max_evts if 0 < args.max_evts <= n_tot_files else n_tot_files
    print(
        "Out of {} events processing {} events with {} workers.\n".format(
            n_tot_files, max_evts, args.num_workers
        )
    )

    # Multiprocess to Build Tracks (Torch Files)
    print("Write Track Candidates as TORCH")
    with Pool(args.num_workers) as p:
        process_fnc = partial(process, **vars(args))
        p.map(process_fnc, all_files[:max_evts])
