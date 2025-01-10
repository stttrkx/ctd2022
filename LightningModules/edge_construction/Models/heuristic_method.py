import os
import logging
import numpy as np
from time import time

from functools import partial
from tqdm.contrib.concurrent import process_map

from ..heuristic_base import HeuristicBase
from ..utils.heuristic_utils import construct_graphs

# TODO: Idea is to use a Heuristic Method for graph construction. Currently, this
# is done in prepare_event() in data_processing since is easy to implement there.


class HeuristicMethod(HeuristicBase):
    """
    Heuristic method for graph construction, we need raw events (CSVs) to build
    graphs (layerwise, all_edges) that work on dataframes. We also need output
    of data_processing where we need to add constructed graph. Finally, we store
    data into output_dir withou splitting, once it works we will split data and
    store it into train, val and test folders (we split by hand in the past).
    """

    def __init__(self, hparams: dict):
        super().__init__(hparams)

        # self.detector_path = self.hparams["detector_path"]

    def prepare_data(self):
        """Preparing dataset"""

        start_time = time()

        # Create the output directory if it does not exist
        logging.info("Writing outputs to " + self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info("Using the TrackMLFeatureStore to process data from CSV files.")

        # TODO: Use loading functions of embedding_utils. Idea is to fetch n_files
        # split it into train_split and then construct graph and finally store it
        # into train, val and test directories using the train_split

        # Find the input files
        all_files = os.listdir(self.input_dir)
        all_events = sorted(
            np.unique([os.path.join(self.input_dir, event[:15]) for event in all_files])
        )[: self.n_files]

        # Split the input files by number of tasks and select my chunk only
        all_events = np.array_split(all_events, self.n_tasks)[self.task]

        # Process input files with a worker pool and progress bar
        # Use process_map() from tqdm instead of mp.Pool from multiprocessing.
        process_func = partial(construct_graphs, **self.hparams)
        process_map(
            process_func,
            all_events,
            max_workers=self.n_workers,
            chunksize=self.chunksize,
        )

        # Print the time taken for feature construction
        end_time = time()
        print(
            f"Feature construction complete. Time taken: {end_time - start_time:f} seconds."
        )
