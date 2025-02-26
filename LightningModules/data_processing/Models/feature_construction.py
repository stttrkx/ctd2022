import os
import yaml
import logging
import numpy as np
import uproot as up
import pandas as pd
import awkward as ak
import fnmatch

from time import time
from functools import partial
from tqdm.contrib.concurrent import process_map

from ..feature_store_base import FeatureStoreBase
from ..utils.trackml_event_utils import prepare_event as trackml_prepare_event
from ..utils.panda_event_utils import prepare_event as panda_prepare_event
from ..utils.root_file_reader import ROOTFileReader
from ..utils.pandaRoot_event_utils import prepare_event as pandaRoot_prepare_event

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class TrackMLFeatureStore(FeatureStoreBase):
    """
    Processing model to convert STT data into files ready for GNN training

    Description:
        This class is used to read data from csv files containing PANDA STT hit and MC truth information.
        This information is then used to create true and input graphs and saved in PyTorch geometry files.
        It is a subclass of FeatureStoreBase which in turn is a subclass of the PyTorch Lighting's LightningDataModule.
    """

    def __init__(self, hparams: dict):
        """
        Initializes the TrackMLFeatureStore class.

        Args:
            hparams (dict): Dictionary containing the hyperparameters for the feature construction / data processing
        """

        # Call the base class (FeatureStoreBase in feature_store_base.py) constructor with the hyperparameters as arguments
        super().__init__(hparams)

        # self.detector_path = self.hparams["detector_path"]

    def prepare_data(self):
        """
        Main function for the feature construction / data processing.

        Description:
            Parallelizes the processing of input files by splitting them into
            evenly sized chunks and processing each chunk in parallel.
        """

        start_time = time()

        # Create the output directory if it does not exist
        logging.info("Writing outputs to " + self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        logging.info("Using the TrackMLFeatureStore to process data from CSV files.")

        # Find the input files
        all_files = os.listdir(self.input_dir)
        all_events = sorted(
            np.unique([os.path.join(self.input_dir, event[:15]) for event in all_files])
        )[: self.n_files]

        # Split the input files by number of tasks and select my chunk only
        all_events = np.array_split(all_events, self.n_tasks)[self.task]

        # Process input files with a worker pool and progress bar
        # Use process_map() from tqdm instead of mp.Pool from multiprocessing.
        process_func = partial(trackml_prepare_event, **self.hparams)
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


class PandaFeatureStore(FeatureStoreBase):
    """
    Class to process ROOT files containing PANDA STT data and save the processed tensors into PyTorch files.
    """

    def __init__(self, hparams: dict) -> None:
        """
        Default constructor for the PandaFeatureStore class.

        Initializes the PandaFeatureStore class by calling the FeatureStoreBase constructor with a dictionary containing the hyperparameters.

        Args:
            hparams (dict): Dictionary containing the hyperparameters for the PANDA data processing.
        """

        # Call the base class (FeatureStoreBase in feature_store_base.py) constructor with the hyperparameters as arguments
        super().__init__(hparams)

    def prepare_data(self) -> None:
        """
        Main method for the PANDA data processing.
        """

        # Start the timer to measure the time taken for feature construction
        start_time = time()

        # Create the output directory if it does not exist yet
        logging.info("Writing outputs to " + self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Check if the input file is a ROOT file by examining the extension
        fileExtension = os.path.splitext(self.input_dir)[1]
        if fileExtension != ".root":
            logging.error(f"Specified input file {self.input_dir} is not a ROOT file!")
            raise Exception("Input file must be a ROOT file.")

        # Open the input ROOT file using the ROOTFileReader class and get the number of events saved in the file
        root_file_reader = ROOTFileReader(self.input_dir)
        total_events = root_file_reader.get_tree_entries()
        logging.info(f"Total number of events in the file: {total_events}")

        # Get the number of events to process
        # If this hyperparameter is not specified, process all events in the file
        if "n_files" not in self.hparams.keys():
            nEvents = total_events
        else:
            nEvents = self.hparams["n_files"]

        logging.info(f"Number of events to process: {nEvents}")

        # make iterable of events
        all_events = range(nEvents)

        # Define a new function by passing the static arguments to the prepare_event function
        process_func = partial(
            panda_prepare_event, file_reader=root_file_reader, **self.hparams
        )

        # Execute the new process_func in parallel for each event withing the all_events iterable
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


class PandaRootFeatureStore(FeatureStoreBase):
    """
    Class to process data from PandaRoot sim and digi files and save the processed tensors into PyTorch files.
    """

    def __init__(self, hparams: dict) -> None:
        """
        Default constructor for the PandaRootFeatureStore class.

        Initializes the PandaRootFeatureStore class by calling the FeatureStoreBase constructor with a dictionary containing the hyperparameters.

        Args:
            hparams (dict): Dictionary containing the hyperparameters for the PandaRoot data processing.
        """

        # Call the base class (FeatureStoreBase in feature_store_base.py) constructor with the hyperparameters as arguments
        super().__init__(hparams)

    def prepare_data(self) -> None:
        """
        Main method for the PandaRoot data processing.
        """

        # Start the timer to measure the time taken for feature construction
        start_time = time()

        # Create the output directory if it does not exist yet
        logging.info("Writing outputs to " + self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Save the STT geometry data from the csv file into a pandas data frame
        stt_geo_df = pd.read_csv(
            "/home/nikin105/mlProject/data/detectorGeometries/tubePos.csv"
        )

        # Create a pandas data frame to save some event wise meta information
        event_info_df = pd.DataFrame(
            columns=[
                "event_id",
                "n_true_edges",
                "n_input_edges",
                "n_true_input_edges",
                "n_false_input_edges",
                "n_hits",
                "n_zero_charge_hits",
                "n_multi_hits",
            ],
            dtype=int,
        )

        # Read the signal signature from the YAML file.
        with open(self.hparams["signal_signature_file"], "r") as file:
            signal_signature = yaml.safe_load(file)

        # Count the number of input sim and digi ROOT files
        num_sim_files = 0
        for filename in os.listdir(self.input_dir + "/sim"):
            if fnmatch.fnmatch(filename, "*_sim.root"):
                num_sim_files += 1
        logging.info(f"Number of sim files: {num_sim_files}")

        num_digi_files = 0
        for filename in os.listdir(self.input_dir + "/digi"):
            if fnmatch.fnmatch(filename, "*_digi.root"):
                num_digi_files += 1
        logging.info(f"Number of digi files: {num_digi_files}")

        # Check if the number of sim and digi files are the same
        if num_sim_files != num_digi_files:
            logging.error(
                f"Number of sim files ({num_sim_files}) and digi files ({num_digi_files}) do not match!"
            )
            raise Exception("Number of sim and digi files must match.")

        # Names of the TBranches in the ROOT files to be processed and their corresponding names in the data frame
        sttPoint_branch_dict = {
            "STTPoint.fX": "tx",
            "STTPoint.fY": "ty",
            "STTPoint.fZ": "tz",
            "STTPoint.fTime": "tT",
            "STTPoint.fPx": "tpx",
            "STTPoint.fPy": "tpy",
            "STTPoint.fPz": "tpz",
            "STTPoint.fTrackID": "particle_id",
            "STTPoint.fX_out_local": "x_out",
            "STTPoint.fY_out_local": "y_out",
            "STTPoint.fX_in_local": "x_in",
            "STTPoint.fY_in_local": "y_in",
        }

        mcTrack_branch_dict = {
            "MCTrack.fStartX": "vx",
            "MCTrack.fStartY": "vy",
            "MCTrack.fStartZ": "vz",
            "MCTrack.fPdgCode": "pdgcode",
            "MCTrack.fProcess": "process_code",
            "MCTrack.fMotherID": "mother_id",
            "MCTrack.fSecondMotherID": "second_mother_id",
        }

        sttHit_branch_dict = {
            "STTHit.fRefIndex": "hit_id",
            "STTHit.fX": "x",
            "STTHit.fY": "y",
            "STTHit.fZ": "z",
            "STTHit.fDetectorID": "volume_id",
            "STTHit.fTubeID": "module_id",
            "STTHit.fIsochrone": "isochrone",
            "STTHit.fDepCharge": "dep_charge",
        }

        # Dictionary containing the keys of the data frame for the different branch types
        key_dict = {
            "sttPoint": [
                sttPoint_branch_dict[sttPoint_key]
                for sttPoint_key in sttPoint_branch_dict.keys()
            ],
            "mcTrack": [
                mcTrack_branch_dict[mcTrack_key]
                for mcTrack_key in mcTrack_branch_dict.keys()
            ],
            "sttHit": [
                sttHit_branch_dict[sttHit_key]
                for sttHit_key in sttHit_branch_dict.keys()
            ],
        }

        # Count the number of events processed
        events_processed = 0

        # Iterate over all files
        for file_num in range(self.n_files):

            print(f"Step {file_num+1} of {self.n_files}")

            # Open the simulation file using uproot
            sim_file_name = (
                self.input_dir
                + "/sim/"
                + self.hparams["prefix"]
                + f"_{file_num}_sim.root:pndsim"
            )
            logging.info(f"Simulation file:\n{sim_file_name}")
            sim_file = up.open(sim_file_name, num_workers=self.n_workers)

            # Create an iterator that contains a chunk of the simulation events
            sim_iterator = sim_file.iterate(
                expressions=list(mcTrack_branch_dict.keys())
                + list(sttPoint_branch_dict.keys()),
                library="pd",
                step_size=self.hparams["events_per_step"],
            )

            digi_file_name = (
                self.input_dir
                + "/digi/"
                + self.hparams["prefix"]
                + f"_{file_num}_digi.root:pndsim"
            )
            logging.info(f"Digitalization file:\n{digi_file_name}")
            digi_file = up.open(digi_file_name, num_workers=self.n_workers)

            # Create an iterator that contains a chunk of the digitalization events
            digi_iterator = digi_file.iterate(
                expressions=sttHit_branch_dict.keys(),
                library="pd",
                step_size=self.hparams["events_per_step"],
            )

            # Iterate over the chunks of events
            for chunk, digi_chunk in zip(sim_iterator, digi_iterator):
                # Rename the columns of the chunks so they are correctly named in the tuples and
                # are consistent with the other implementations
                chunk = chunk.rename(columns=mcTrack_branch_dict)
                chunk = chunk.rename(columns=sttPoint_branch_dict)
                digi_chunk = digi_chunk.rename(columns=sttHit_branch_dict)

                logging.debug(f"Simulation chunk:\n{chunk}")
                logging.debug(f"Digitalization chunk:\n{digi_chunk}")

                # Make an array with the event ids for the current chunk
                last_event_num = events_processed + len(chunk)
                event_ids = np.arange(events_processed, last_event_num, dtype=int)

                # Combine the simulation and digitalization chunks into a single chunk
                chunk = pd.concat([chunk, digi_chunk], axis=1)
                chunk = chunk.assign(event_id=event_ids)
                logging.debug(f"Full chunk:\n{chunk}")
                del digi_chunk

                chunk = chunk[chunk.event_id < self.hparams["n_events"]]

                # Create a progress bar for the current chunk
                progress_bar = tqdm(total=chunk.shape[0])

                # Define a new function by passing the static arguments to the prepare_event function
                process_func = partial(
                    pandaRoot_prepare_event,
                    key_dict=key_dict,
                    signal_signatures=signal_signature,
                    stt_geo=stt_geo_df,
                    progress_bar=progress_bar,
                    **self.hparams,
                )

                # If there is only one worker don't use the ThreadPoolExecutor
                if self.n_workers == 1:
                    # Execute the process_func for each event in the chunk
                    for event in chunk.itertuples():
                        event_info_df = pd.concat(
                            [
                                event_info_df,
                                pd.DataFrame(
                                    [process_func(event)], columns=event_info_df.columns
                                ),
                            ],
                            ignore_index=True,
                        )
                else:
                    # Execute the process_func in parallel for each event
                    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                        event_info_df = pd.concat(
                            [
                                event_info_df,
                                pd.DataFrame(
                                    list(
                                        executor.map(process_func, chunk.itertuples())
                                    ),
                                    columns=event_info_df.columns,
                                ),
                            ],
                            ignore_index=True,
                        )

                # Close the progress bar
                progress_bar.close()

                # Update the number of events processed
                events_processed += len(chunk)

            # Close the ROOT files
            sim_file.close()
            digi_file.close()

        # End the timer and print the time taken for feature construction
        end_time = time()
        print(
            f"Feature construction complete. Time taken: {end_time - start_time:f} seconds."
        )

        # Print some information about the edge construction
        n_true_edges = event_info_df["n_true_edges"].sum()
        n_input_edges = event_info_df["n_input_edges"].sum()
        n_true_input_edges = event_info_df["n_true_input_edges"].sum()
        n_false_input_edges = event_info_df["n_false_input_edges"].sum()
        n_missing_true_edges = n_true_edges - n_true_input_edges

        print("Total class imbalance:")
        print(f"Percentage of true edges: {n_true_input_edges/n_input_edges*100:.2f}%")
        print(
            f"Percentage of false edges: {n_false_input_edges/n_input_edges*100:.2f}%"
        )

        if n_missing_true_edges != 0:
            print(f"Total number of missing true edges: {n_missing_true_edges}")
            print(
                f"Total input edge construction efficiency: {n_true_input_edges/n_true_edges*100:.2f}%"
            )
            event_info_df["missing_true_edges"] = (
                event_info_df["n_true_edges"] - event_info_df["n_true_input_edges"]
            )
            print(
                f"Event with the most missing edges: {event_info_df['event_id'].iloc[event_info_df['missing_true_edges'].idxmax()]}"
            )

        # Save the event information to a HDF5 file
        event_info_df.to_hdf(
            os.path.join(self.output_dir, "event_info.h5"),
            key="event_info_df",
            mode="w",
        )
