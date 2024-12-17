#!/usr/bin/env python
# coding: utf-8

import os
import logging
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import to_networkx, from_networkx
from itertools import chain
from functools import partial
from .utils_fit import pairwise, poly_fit_phi


def find_next_hits(G, pp, used_hits, th=0.1, th_re=0.8, feature_name="solution"):
    """G is the graph, path is previous hits."""

    nbrs = list(set(G.neighbors(pp)).difference(set(used_hits)))
    if len(nbrs) < 1:
        return None

    # weights = [G.edges[(pp, i)][feature_name][0] for i in nbrs]

    # FIXME (done): In the DiGraph from PyD::to_networkx(PyG::Data,...), each
    # edge has a score and truth (y_pid), which is interpretted as the
    # weight of an edge. From this DiGraph structure one can find it as:

    weights = [G.edges[(pp, i)][feature_name] for i in nbrs]

    if max(weights) < th:
        return None

    sorted_idx = list(reversed(np.argsort(weights)))
    next_hits = [nbrs[sorted_idx[0]]]

    if len(sorted_idx) > 1:
        for ii in range(1, len(sorted_idx)):
            idx = sorted_idx[ii]
            w = weights[idx]
            if w > th_re:
                next_hits.append(nbrs[idx])
            else:
                break

    return next_hits


def build_roads(G, ss, next_hit_fn, used_hits):
    """
    next_hit_fn: a function return next hits, could be find_next_hits
    """
    # get started
    next_hits = next_hit_fn(G, ss, used_hits)
    if next_hits is None:
        return [(ss, None)]
    path = []
    for hit in next_hits:
        path.append((ss, hit))

    while True:
        new_path = []
        is_all_none = True
        for pp in path:
            if pp[-1] is not None:
                is_all_none = False
                break
        if is_all_none:
            break

        for pp in path:
            start = pp[-1]
            if start is None:
                new_path.append(pp)
                continue

            used_hits_cc = np.unique(used_hits + list(pp))
            next_hits = next_hit_fn(G, pp[-1], used_hits_cc)
            if next_hits is None:
                new_path.append(pp + (None,))
            else:
                for hit in next_hits:
                    new_path.append(pp + (hit,))

        path = new_path
    return path


def fit_road(G, road):
    """use a linear function to fit phi as a function of z."""
    road_chi2 = []
    for path in road:
        z = np.array(
            [G.nodes[i]["x"][2] for i in path[:-1]]
        )  # ADAK: G.node (v1.x) to G.nodes (v2.x)
        phi = np.array([G.nodes[i]["x"][1] for i in path[:-1]])  # ADAK: 'pos' to 'x'
        if len(z) > 1:
            _, _, diff = poly_fit_phi(z, phi)
            road_chi2.append(np.sum(diff) / len(z))
        else:
            road_chi2.append(1)

    return road_chi2


def chose_a_road(road, diff):
    res = road[0]
    # only if another road has small difference in phi-fit
    # and longer than the first one, it is used.
    for i in range(1, len(road)):
        if diff[i] <= diff[0] and len(road[i]) > len(res):
            res = road[i]

    return res


def get_tracks(G, th=0.1, th_re=0.8, feature_name="scores", with_fit=True):
    """
    Don't use nx.MultiGraphs
    """
    used_nodes = []
    sub_graphs = []
    next_hit_fn = partial(find_next_hits, th=th, th_re=th_re, feature_name=feature_name)
    for node in G.nodes():
        if node in used_nodes:  # TODO: ADAK: remove this condition to allow shared hits
            continue
        road = build_roads(G, node, next_hit_fn, used_nodes)
        diff = fit_road(G, road) if with_fit else [0.0] * len(road)
        a_road = chose_a_road(road, diff)

        if len(a_road) < 3:
            used_nodes.append(node)
            sub_graphs.append(G.subgraph([node]))
            continue

        a_track = list(pairwise(a_road[:-1]))
        sub = G.edge_subgraph(a_track)
        sub_graphs.append(sub)
        used_nodes += list(sub.nodes())

    return sub_graphs


def wrangler_labelling(input_file, output_dir, edge_cut=0.5, **kwargs):
    """Find tracks using a Walktrhough method..."""

    try:
        output_file = os.path.join(output_dir, os.path.split(input_file)[-1])
        if not os.path.exists(output_file) or kwargs["overwrite"]:

            logging.info("Preparing event {}".format(output_file))
            graph = torch.load(input_file, map_location="cpu")

            # edge scores
            scores = graph.scores

            # half the length, gnn gives scores for bidirected graphs
            scores = scores[: graph.edge_index.shape[1]]

            # apply edge score cut
            edge_mask = scores > edge_cut

            # Convert to sparse scipy array
            new_graph = graph.clone()
            new_graph.edge_index = new_graph.edge_index[:, edge_mask]
            new_graph.scores = new_graph.scores[edge_mask]

            # Convert to networkx graph
            G = to_networkx(
                new_graph,
                node_attrs=["x"],
                edge_attrs=["scores", "y_pid"],
                to_undirected=False,
            )

            # Build tracks
            paths = get_tracks(
                G, th=0.1, th_re=0.8, feature_name="scores", with_fit=False
            )

            track_df = pd.DataFrame(
                {
                    "hit_id": list(chain.from_iterable(paths)),
                    "track_id": list(
                        chain.from_iterable([[i] * len(p) for i, p in enumerate(paths)])
                    ),
                }
            )

            # list subgraphs as pyg data
            # pyg_graph = [from_networkx(graph) for graph in paths]

            with open(output_file, "wb") as pickle_file:
                torch.save(track_df, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)
