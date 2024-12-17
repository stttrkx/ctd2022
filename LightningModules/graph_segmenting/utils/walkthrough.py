# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Original: https://github.com/GNN4ITkTeam/CommonFramework

# 3rd party imports
import os
import logging
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sps
from tqdm import tqdm
from multiprocessing import Pool
from torch_geometric.utils import to_networkx
import networkx as nx
from itertools import chain
from functools import partial

# Local imports
from torch_geometric.utils import to_scipy_sparse_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_all_paths(start, G=None, ending_nodes=None):
    return list(
        chain.from_iterable(
            [
                list(nx.all_simple_paths(G, start, end))
                for end in ending_nodes
                if nx.has_path(G, start, end)
            ]
        )
    )


def find_shortest_paths(start, G=None, ending_nodes=None):
    return [
        nx.shortest_path(G, start, end)
        for end in ending_nodes
        if nx.has_path(G, start, end)
    ]


def walkthrough_labelling(input_file, output_dir, edge_cut=0.5, **kwargs):
    """
    Given a set of scored graphs, and a score cut, build tracks from graphs by:
    1. Applying the score cut to the graph
    2. Running walkthrough path method as a partial method
    """
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
            G = to_networkx(new_graph, to_undirected=False)
            G.remove_nodes_from(list(nx.isolates(G)))

            starting_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
            ending_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

            workers = 8  # TODO: Remove this hardcoded value

            # Make partial method for multiprocessing
            find_paths_partial = partial(
                find_shortest_paths, G=G, ending_nodes=ending_nodes
            )

            # Run multiprocessing
            with Pool(workers) as p:
                paths = list(p.map(find_paths_partial, starting_nodes))

            track_df = pd.DataFrame(
                {
                    "hit_id": list(chain.from_iterable(paths)),
                    "track_id": list(
                        chain.from_iterable([[i] * len(p) for i, p in enumerate(paths)])
                    ),
                }
            )

            # Remove duplicates on hit_id: TODO: In very near future, handle multiple tracks through the same hit!
            track_df = track_df.drop_duplicates(subset="hit_id")

            hit_id = track_df.hit_id
            track_id = track_df.track_id

            track_id_tensor = torch.ones(len(graph.x), dtype=torch.long) * -1
            track_id_tensor[hit_id.values] = torch.from_numpy(track_id.values)

            graph.labels = track_id_tensor

            # Save Files
            with open(output_file, "wb") as pickle_file:
                torch.save(graph, pickle_file)

        else:
            logging.info("{} already exists".format(output_file))

    except Exception as inst:
        print("File:", input_file, "had exception", inst)
