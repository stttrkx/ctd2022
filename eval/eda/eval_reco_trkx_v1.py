#!/usr/bin/env python
# coding: utf-8

"""Modified version of 'eval_reco_trkx.py' (runs after 'tracks_from_gnn.py') script from
gnn4itk repo. The code breakdown of the script is given in 'stt6_eval.ipynb' notebook."""

import os
import glob
import torch
import numpy as np
import pandas as pd
from typing import Any

device = "cuda" if torch.cuda.is_available() else "cpu"


class SttTorchDataReader(object):
    """Torch Geometric Data Reader from an Input Directory."""

    def __init__(self, input_dir: str):
        """Initialize Instance Variables in Constructor"""

        self.path = input_dir
        all_files = sorted(glob.glob(os.path.join(input_dir, "*")))
        self.nevts = len(all_files)
        self.all_evtids = [os.path.basename(x) for x in all_files]

    def read(self, evtid: int = None):
        """Read an Event from the Input Directory."""
        event_fname = os.path.join(self.path, "{}".format(evtid))
        event = torch.load(event_fname, map_location=device)
        return event

    def __call__(self, evtid: int, *args: Any, **kwds: Any) -> Any:
        return self.read(evtid)


def evaluate_reco_tracks(
    truth_df: pd.DataFrame,
    reco_df: pd.DataFrame,
    particles_df: pd.DataFrame,
    min_hits_truth: int = 9,
    min_hits_reco: int = 5,
    min_pt: float = 1.0,
    max_pt: float = 1.5,
    frac_reco_matched: float = 0.5,
    frac_truth_matched=0.5,
    **kwargs
):
    """
    Args:
        truth_df: a dataframe with columns of ['hit_id', 'particle_id']
        reco_df: a dataframe with columns of ['hit_id', 'track_id']
        particles_df: a dataframe with columns of
            ['particle_id', 'pt', 'eta', 'radius', 'vz'].
            where radius = sqrt(vx**2 + vy**2) and
            ['vx', 'vy', 'vz'] are the production vertex of the particle
        min_hits_truth: minimum number of hits for truth tracks
        min_hits_reco:  minimum number of hits for reconstructed tracks
        min_pt: minimum pT to filter out
        max_pt: maximum pT to filter out
        frac_reco_matched: frac of reco tracks matched ??? (ADAK)
        frac_truth_matched: frac of true tracks matched ??? (ADAK)

    Returns:
        A tuple of (
            num_true_tracks: int, number of true tracks
            num_reco_tracks: int, number of reconstructed tracks
            n_matched_reco_tracks: int, number of reconstructed tracks matched to true tracks
            matched_pids: np.narray, a list of particle IDs matched by reconstructed tracks
        )
    """
    # just in case particle_id == 0 included in truth.
    if "particle_id" in truth.columns:
        truth = truth[truth.particle_id > 0]

    # get number of spacepoints in each reconstructed tracks
    n_reco_hits = (
        reconstructed.track_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "track_id", "track_id": "n_reco_hits"})
    )

    # only tracks with a minimum number of spacepoints are considered
    n_reco_hits = n_reco_hits[n_reco_hits.n_reco_hits >= min_hits_reco]
    reconstructed = reconstructed[
        reconstructed.track_id.isin(n_reco_hits.track_id.values)
    ]

    # get number of spacepoints in each particle
    hits = truth.merge(particles, on="particle_id", how="left")
    n_true_hits = (
        hits.particle_id.value_counts(sort=False)
        .reset_index()
        .rename(columns={"index": "particle_id", "particle_id": "n_true_hits"})
    )

    # only particles leaves at least min_hits_truth spacepoints
    # and with pT >= min_pt are considered.
    particles = particles.merge(n_true_hits, on=["particle_id"], how="left")

    is_trackable = particles.n_true_hits >= min_hits_truth

    # event has 3 columnes [track_id, particle_id, hit_id]
    event = pd.merge(reconstructed, truth, on=["hit_id"], how="left")

    # n_common_hits and n_shared should be exactly the same
    # for a specific track id and particle id

    # Each track_id will be assigned to multiple particles.
    # To determine which particle the track candidate is matched to,
    # we use the particle id that yields a maximum value of n_common_hits / n_reco_hits,
    # which means the majority of the spacepoints associated with the reconstructed
    # track candidate comes from that true track.
    # However, the other way may not be true.
    reco_matching = (
        event.groupby(["track_id", "particle_id"])
        .size()
        .reset_index()
        .rename(columns={0: "n_common_hits"})
    )

    # Each particle will be assigned to multiple reconstructed tracks
    truth_matching = (
        event.groupby(["particle_id", "track_id"])
        .size()
        .reset_index()
        .rename(columns={0: "n_shared"})
    )

    # add number of hits to each of the maching dataframe
    reco_matching = reco_matching.merge(n_reco_hits, on=["track_id"], how="left")
    truth_matching = truth_matching.merge(n_true_hits, on=["particle_id"], how="left")

    # calculate matching fraction
    reco_matching = reco_matching.assign(
        purity_reco=np.true_divide(
            reco_matching.n_common_hits, reco_matching.n_reco_hits
        )
    )
    truth_matching = truth_matching.assign(
        purity_true=np.true_divide(truth_matching.n_shared, truth_matching.n_true_hits)
    )

    # select the best match
    reco_matching["purity_reco_max"] = reco_matching.groupby("track_id")[
        "purity_reco"
    ].transform(max)
    truth_matching["purity_true_max"] = truth_matching.groupby("track_id")[
        "purity_true"
    ].transform(max)

    matched_reco_tracks = reco_matching[
        (reco_matching.purity_reco_max >= frac_reco_matched)
        & (reco_matching.purity_reco == reco_matching.purity_reco_max)
    ]

    matched_true_particles = truth_matching[
        (truth_matching.purity_true_max >= frac_truth_matched)
        & (truth_matching.purity_true == truth_matching.purity_true_max)
    ]

    # now, let's combine the two majority criteria
    # reconstructed tracks must be in both matched dataframe
    # and the so matched particle should be the same
    # in this way, each track should be only assigned
    combined_match = matched_true_particles.merge(
        matched_reco_tracks, on=["track_id", "particle_id"], how="inner"
    )

    n_reco_tracks = n_reco_hits.shape[0]
    n_true_tracks = particles.shape[0]

    # For GNN, there are non-negaliable cases where GNN-based
    # track candidates are matched to particles not considered as interesting.
    # which means there are paticles in matched_pids that do not exist in particles.
    matched_pids = np.unique(combined_match.particle_id)

    is_matched = particles.particle_id.isin(matched_pids).values
    n_matched_particles = np.sum(is_matched)

    n_matched_tracks = reco_matching[
        reco_matching.purity_reco >= frac_reco_matched
    ].shape[0]
    n_matched_tracks_poi = reco_matching[
        (reco_matching.purity_reco >= frac_reco_matched)
        & (reco_matching.particle_id.isin(particles.particle_id.values))
    ].shape[0]
    # print(n_matched_tracks_poi, n_matched_tracks)

    # num_particles_matched_to = reco_matched.groupby("particle_id")['track_id']\
    #     .count().reset_index().rename(columns={"track_id": "n_tracks_matched"})
    # n_duplicated_tracks = num_particles_matched_to.shape[0]
    n_duplicated_tracks = n_matched_tracks_poi - n_matched_particles

    particles = particles.assign(is_matched=is_matched, is_trackable=is_trackable)

    return (
        n_true_tracks,
        n_reco_tracks,
        n_matched_particles,
        n_matched_tracks,
        n_duplicated_tracks,
        n_matched_tracks_poi,
        particles,
    )


def run_one_evt(evtid, raw_trkx_data_reader, reco_trkx_data_reader, **kwargs):
    print("Running {}".format(evtid))

    # access torch data using reader's
    raw_trkx_data = raw_trkx_data_reader(evtid)
    reco_trkx_data = reco_trkx_data_reader(evtid)

    # create truth, particles dataframes from torch data
    _truth = pd.DataFrame(
        {
            "hit_id": raw_trkx_data.hid.numpy(),
            "particle_id": raw_trkx_data.pid.int().numpy(),
        },
        columns=["hit_id", "particle_id"],
    )

    _particles = pd.DataFrame(
        {
            "particle_id": raw_trkx_data.pid.int().numpy(),
            "pt": raw_trkx_data.pt.numpy(),
            "vx": raw_trkx_data.vertex[:, 0].numpy(),
            "vy": raw_trkx_data.vertex[:, 1].numpy(),
            "vz": raw_trkx_data.vertex[:, 2].numpy(),
            "q": raw_trkx_data.charge.numpy(),
            "pdgcode": raw_trkx_data.pdgcode.numpy(),
            "ptheta": raw_trkx_data.ptheta.numpy(),
            "peta": raw_trkx_data.peta.numpy(),
            "pphi": raw_trkx_data.pphi.numpy(),
        },
        columns=[
            "particle_id",
            "pt",
            "vx",
            "vy",
            "vz",
            "q",
            "pdgcode",
            "ptheta",
            "peta",
            "pphi",
        ],
    ).drop_duplicates(subset=["particle_id"])

    results = evaluate_reco_tracks(_truth, reco_trkx_data, _particles, **kwargs)

    return results[:-1] + (results[-1].assign(evtid=evtid),)


# %%
if __name__ == "__main__":
    import time
    import argparse
    from multiprocessing import Pool
    from functools import partial

    parser = argparse.ArgumentParser(description="Evaluating tracking reconstruction")
    add_arg = parser.add_argument
    add_arg("--reco-tracks-path", help="path to reconstructed tracks", required=True)
    add_arg(
        "--raw-tracks-path",
        help="path to raw tracking data (for truth info)",
        required=True,
    )
    add_arg("--outname", help="output name without postfix", required=True)
    add_arg("--max-evts", help="maximum number of events", type=int, default=1)
    add_arg(
        "-e", "--event-id", help="evaluate a particular event", type=int, default=None
    )
    add_arg(
        "-f", "--force", help="force to over write existing file", action="store_true"
    )
    add_arg("--num-workers", help="number of workers", default=1, type=int)

    add_arg(
        "--min-hits-truth",
        help="minimum number of hits in a truth track",
        default=7,
        type=int,
    )
    add_arg(
        "--min-hits-reco",
        help="minimum number of hits in a reconstructed track",
        default=4,
        type=int,
    )
    add_arg("--min-pt", help="minimum pT of true track", type=float, default=1.0)
    add_arg("--max-pt", help="maximum pT of true track", type=float, default=10.0)
    add_arg(
        "--frac-reco-matched",
        help="fraction of matched hits over total hits in a reco track",
        default=0.5,
        type=float,
    )
    add_arg(
        "--frac-truth-matched",
        help="fraction of matched hits over total hits in a truth track",
        default=0.5,
        type=float,
    )

    args = parser.parse_args()
    reco_track_path = args.reco_tracks_path
    num_workers = args.num_workers
    outname = args.outname
    outdir = os.path.dirname(os.path.abspath(outname))
    os.makedirs(outdir, exist_ok=True)

    # read reconstructed tracks
    reco_trkx_reader = SttTorchDataReader(reco_track_path)

    n_tot_files = reco_trkx_reader.nevts
    all_evtids = reco_trkx_reader.all_evtids
    max_evts = args.max_evts if 0 < args.max_evts <= n_tot_files else n_tot_files

    print(
        "Out of {} events processing {} events with {} workers".format(
            n_tot_files, max_evts, args.num_workers
        )
    )
    print("Output directory:", outdir)

    # read raw Torch/CSV files to get truth information
    raw_trkx_reader = SttTorchDataReader(args.raw_tracks_path)

    out_array = "{}_particles.h5".format(outname)
    if os.path.exists(out_array) and not args.force:
        print("{} is there, use -f to overwrite the file".format(out_array))
        exit(1)

    if not args.event_id:
        if num_workers > 1:
            with Pool(num_workers) as p:
                fnc = partial(
                    run_one_evt,
                    raw_trkx_reader=raw_trkx_reader,
                    reco_trkx_reader=reco_trkx_reader,
                    **vars(args)
                )

                res = p.map(fnc, all_evtids[:max_evts])
        else:
            res = [
                run_one_evt(evtid, raw_trkx_reader, reco_trkx_reader, **vars(args))
                for evtid in all_evtids[:max_evts]
            ]

        # merge results from each process
        n_true_tracks = sum([x[0] for x in res])
        n_reco_tracks = sum([x[1] for x in res])
        n_matched_true_tracks = sum([x[2] for x in res])
        n_matched_reco_tracks = sum([x[3] for x in res])
        n_duplicated_reco_tracks = sum([x[4] for x in res])
        n_matched_reco_tracks_poi = sum([x[5] for x in res])
        particles = pd.concat([x[-1] for x in res], axis=0)
    else:
        (
            n_true_tracks,
            n_reco_tracks,
            n_matched_true_tracks,
            n_matched_reco_tracks,
            n_duplicated_reco_tracks,
            n_matched_reco_tracks_poi,
            particles,
        ) = run_one_evt(args.event_id, raw_trkx_reader, reco_trkx_reader, **vars(args))

    with pd.HDFStore(out_array, "w") as f:
        f["data"] = particles

    out_sum = "{}_summary.txt".format(outname)
    ctime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    summary = [
        "".join(["-"] * 50),
        "                   Run Time: {:>10}".format(ctime),
        "       Reconstructed tracks: {:>10}".format(os.path.abspath(reco_track_path)),
        "                # of events: {:>10}".format(
            max_evts if not args.event_id else 1
        ),
        "               Truth tracks: {:>10}".format(n_true_tracks),
        "       Truth tracks matched: {:>10}".format(n_matched_true_tracks),
        "       Reconstructed tracks: {:>10}".format(n_reco_tracks),
        "       Reco. tracks matched: {:>10}".format(n_matched_reco_tracks),
        "Reco. tracks matched to POI: {:>10}".format(n_matched_reco_tracks_poi),
        "    Reco. tracks duplicated: {:>10}".format(n_duplicated_reco_tracks),
        "        Tracking Efficiency: {:>10.4f}%".format(
            100 * n_matched_true_tracks / n_true_tracks
        ),
        "Tracking Efficiency (tech.): {:>10.4f}%".format(
            100 * n_matched_reco_tracks / n_true_tracks
        ),  # FIXME: Please confirm if its correct (I added myself here)
        "               Track Purity: {:>10.4f}%".format(
            100 * n_matched_reco_tracks / n_reco_tracks
        ),  # FIXME: n_matched_reco_tracks < n_reco_tracks, if not then
        "                  Fake Rate: {:>10.4f}%".format(
            100 - 100 * n_matched_reco_tracks / n_reco_tracks
        ),  # one get prutiry > 100% as a result Fake Rate goes -ve
        "           Duplication Rate: {:>10.4f}%".format(
            100 * n_duplicated_reco_tracks / n_reco_tracks
        ),
    ]

    with open(out_sum, "a") as f:
        f.write("\n".join(summary))
        f.write("\n")
