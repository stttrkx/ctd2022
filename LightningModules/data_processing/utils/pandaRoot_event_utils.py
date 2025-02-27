import os
import logging
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch_geometric.data import Data
from .heuristic_utils import (
    get_all_edges,
    graph_intersection,
    get_layerwise_graph,
    get_time_ordered_true_edges,
    get_layerwise_graph_v2,
)
from .particle_utils import is_signal_particle, get_process_ids, get_all_mother_ids


def process_mcTracks(event: pd.Series, signal_signatures: list) -> pd.DataFrame:
    """
    Process an event containing the mcTrack information and return a DataFrame containing the processed information.

    Args:
        event (pd.Series): Event information containing the mcTrack information.
        signal_signatures (list): Lists containing the PDG MC IDs and VMC process codes of the signal particles.

    Returns:
        pd.DataFrame: DataFrame containing the processed mcTrack information.
    """

    # Create a dictionary to store the processed mcTrack information.
    mcTrack_dict = {}

    # Get the mother ids of all particles.
    mother_ids = get_all_mother_ids(
        mother_ids=event["mother_id"],
        second_mother_ids=event["second_mother_id"],
    )

    # Get the track ids of particles that leave a signal in the STT
    # and save them into the dictionary.
    mcTrack_dict["particle_id"] = np.unique(np.array(event["particle_id"]))

    # Initialize the "is_signal" column of the dictionary with an empty bool array.
    mcTrack_dict["primary"] = np.empty(len(mcTrack_dict["particle_id"]), dtype=bool)

    # mcTrack keys that should be saved and processed.
    mcTrack_keys = [
        "vx",
        "vy",
        "vz",
        "pdgcode",
    ]

    # Iterate over the specified keys and save the particle information of
    # the ones leaving hits in the STT into the dictionary.
    for key in mcTrack_keys:
        mcTrack_dict[key] = event[key][mcTrack_dict["particle_id"]]

    # Iterate over all unique track ids and get the particle wise information.
    particle_num = 0
    for particle_id in mcTrack_dict["particle_id"]:
        # Get the PDG MC IDs and VMC process codes of the particle leaving the track
        # and all its mother particles.
        mc_ids, process_codes = get_process_ids(
            process_ids=event["process_code"],
            mother_ids=mother_ids,
            pdg_ids=event["pdgcode"],
            particle_id=particle_id,
        )
        # Check if the particle is a signal particle.
        mcTrack_dict["primary"][particle_num] = is_signal_particle(
            process_mc_ids=mc_ids,
            process_ids=process_codes,
            signal_mc_ids=signal_signatures["particle_ids"],
            signal_process_ids=signal_signatures["process_codes"],
        )
        particle_num += 1

    # Create a pandas DataFrame from the mcTrack dictionary.
    mcTrack_df = pd.DataFrame(mcTrack_dict)

    # Clean up
    del mcTrack_dict
    del mother_ids

    # Return the processed mcTrack DataFrame.
    return mcTrack_df


def process_sttPoints(event: pd.Series, key_dict: dict) -> pd.DataFrame:
    """
    Process an event containing the sttPoint information and return a DataFrame containing the processed information.

    Args:
        event (pd.Series): Event information containing the sttPoint information.
        key_dict (dict): Dictionary containing the keys of the event series corresponding the different branch types.

    Returns:
        pd.DataFrame: DataFrame containing the processed sttPoint information.
    """

    # Create a dictionary to store the processed sttPoint information.
    sttP_dict = {}

    # Iterate over the specified keys and save the particle information of
    # the ones leaving hits in the STT into the dictionary.
    for key in key_dict["sttPoint"]:
        sttP_dict[key] = event[key]

    # Add the hit ids to the dictionary.
    sttP_dict["hit_id"] = np.arange(len(sttP_dict[key_dict["sttPoint"][0]]))

    # Calculate the transverse momentum.
    sttP_dict["ppt"] = np.sqrt(sttP_dict["tpx"] ** 2 + sttP_dict["tpy"] ** 2)

    # Calculate the polar angle theta.
    sttP_dict["ptheta"] = np.arctan2(sttP_dict["ppt"], sttP_dict["tpz"])

    # Calculate the azimuthal angle phi.
    sttP_dict["pphi"] = np.arctan2(sttP_dict["tpy"], sttP_dict["tpx"])

    # Calculate the pseudorapidity eta.
    sttP_dict["peta"] = -np.log(np.tan(sttP_dict["ptheta"] / 2.0))

    # Calculate the local radius of the entering and exiting points of a track in a tube.
    sttP_dict["r_in"] = np.sqrt(sttP_dict["x_in"] ** 2 + sttP_dict["y_in"] ** 2)
    sttP_dict["r_out"] = np.sqrt(sttP_dict["x_out"] ** 2 + sttP_dict["y_out"] ** 2)

    # Create a pandas DataFrame from the sttPoint dictionary.
    sttP_df = pd.DataFrame(sttP_dict)

    # Clean up
    del sttP_dict

    # Return the processed sttPoint DataFrame.
    return sttP_df


def process_sttHits(
    event: pd.Series, key_dict: dict, stt_geo: pd.DataFrame
) -> pd.DataFrame:
    """
    Process an event containing the sttHit information and return a DataFrame containing the processed information.

    Args:
        event (pd.Series): Event information containing the sttHit information.
        key_dict (dict): Dictionary containing the keys of the event series corresponding the different branch types.
        stt_geo (pd.DataFrame): Pandas DataFrame containing the STT geometry information.

    Returns:
        pd.DataFrame: DataFrame containing the processed sttHit information.
    """

    # Create a dictionary to store the processed sttHit information.
    sttH_dict = {}

    # Iterate over the specified keys and save the particle information of
    # the ones leaving hits in the STT into the dictionary.
    for key in key_dict["sttHit"]:
        sttH_dict[key] = event[key]

    # Create empty columns for the layer_id, sector_id, and skewed information.
    sttH_dict["layer_id"] = np.empty(len(sttH_dict[key_dict["sttHit"][0]]), dtype=int)
    sttH_dict["sector_id"] = np.empty(len(sttH_dict[key_dict["sttHit"][0]]), dtype=int)
    sttH_dict["skewed"] = np.empty(len(sttH_dict[key_dict["sttHit"][0]]), dtype=int)

    # Iterate over all hit ids to get the layer, sector and skewness information for each hit.
    hit_num = 0
    for tube_id in sttH_dict["module_id"]:
        sttH_dict["layer_id"][hit_num] = stt_geo["layerID"][tube_id - 1]
        sttH_dict["sector_id"][hit_num] = stt_geo["sectorID"][tube_id - 1]
        sttH_dict["skewed"][hit_num] = stt_geo["skewed"][tube_id - 1]
        hit_num += 1

    # Calculate the transverse distance (r), azimuthal angle (phi), polar angle (theta), and pseudo-rapidity (eta)
    sttH_dict["r"] = np.sqrt(sttH_dict["x"] ** 2 + sttH_dict["y"] ** 2)
    sttH_dict["phi"] = np.arctan2(sttH_dict["y"], sttH_dict["x"])  # Azimuthal angle
    sttH_dict["theta"] = np.arccos(
        sttH_dict["z"]
        / np.sqrt(sttH_dict["x"] ** 2 + sttH_dict["y"] ** 2 + sttH_dict["z"] ** 2)
    )
    sttH_dict["eta"] = -np.log(np.tan(sttH_dict["theta"] / 2.0))

    # Create a pandas DataFrame from the sttHit dictionary.
    sttH_df = pd.DataFrame(sttH_dict)

    # Clean up
    del sttH_dict

    # Return the processed sttHit DataFrame.
    return sttH_df


def prepare_event(
    event: pd.Series,
    key_dict: dict,
    signal_signatures: list,
    stt_geo: pd.DataFrame,
    output_dir: str,
    progress_bar: tqdm,
    overwrite: bool,
    input_edge_method: str,
    min_hits: int,
    **kwargs,
) -> list:
    """
    _summary_

    Args:
        event (pd.Series): Pandas Series containing the event information.
        key_dict (dict): Dictionary containing the keys of the different data frames.
        signal_signatures (list): Lists containing the PDG MC IDs and VMC process codes of the signal particles.
        stt_geo (pd.DataFrame): Pandas DataFrame containing the STT geometry information.
        output_dir (str): Directory where the PyTorch files will be saved.
        progress_bar (tqdm): Progress bar object to track the processing of the events.
        overwrite (bool): Flag to overwrite the PyTorch files if they already exist.
        input_edge_method (str): Method to construct the input edges. Can be "all" or "layerwise".
        min_hits (int): Minimum number of hits required in an event to be processed.

    Returns:
        list: Meta information of the event (currently mostly number of true/false edges).
    """

    # Convert the tuple to a dictionary.
    event = event._asdict()

    # Get the event id.
    event_id = event["event_id"]

    # Prepare the output filename and check if it already exists.
    output_filename = f"{output_dir}/{event_id}"
    if not os.path.exists(output_filename) or overwrite:
        logging.info(f"Writing into {output_filename}")
    else:
        logging.warning(
            f"File {output_filename} already exists! Skipping event {event_id}..."
        )
        return np.array(
            [
                event_id,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=int,
        )

    # Make create the processed data frame from the processed mcTrack information.
    processed_df = process_mcTracks(event, signal_signatures)

    # Add the sttPoint information to the processed data frame.
    processed_df = pd.merge(
        process_sttPoints(event, key_dict), processed_df, on="particle_id"
    )

    # Add the sttHit information to the processed data frame.
    processed_df = pd.merge(
        process_sttHits(event, key_dict, stt_geo), processed_df, on="hit_id"
    )

    # skip noise hits.
    if not kwargs["noise"]:
        processed_df.query("primary==1", inplace=True)

    # skip skewed tubes
    if not kwargs["skewed"]:
        processed_df.query("skewed==0", inplace=True)

    # count the number of hits with 0 deposited charge
    n_zero_charge = len(processed_df.query("dep_charge==0"))

    if kwargs["remove_shell_hits"]:
        # skip hits with 0 deposited charge
        processed_df.query("dep_charge!=0", inplace=True)
    else:
        # set the isochrone to the tube radius (0.5cm) if the deposited charge is 0
        processed_df.loc[processed_df["dep_charge"] == 0, "isochrone"] = 0.5

    if kwargs["merge_wire_hits"]:
        # Get the row numbers of the hits that correspond to an incoming particle hitting the wire.
        # These hits will have an outgoing local radius of less then tube (0.5cm) and the wire radius (0.001cm).
        tol = 1e-10  # floating point tolerance
        i_wire_hit = processed_df.query("r_out <= 0.001 + @tol").index

        # In these cases the next hit should correspond to the particle leaving the wire as an ingoing hit.
        i_wire_hit_next = i_wire_hit + 1

        # Test this hypothesis
        if not all(processed_df.loc[i_wire_hit_next, "r_in"] <= 0.001 + tol):
            logging.error(
                "The next hit after the wire hit does not correspond to the wire hit!"
            )
            exit(1)

        # Now set the isochrones of the wire hits to 0 and throw away the second hit.
        processed_df.loc[i_wire_hit, "isochrone"] = 0.0
        processed_df.drop(i_wire_hit_next, inplace=True)

    # count the number of times a particle (particle_id) leaves multiple hits in the same tube (module_id)
    duplicate_hits = processed_df[
        processed_df.duplicated(subset=["particle_id", "module_id"], keep="first")
    ]
    n_multi_hits = len(duplicate_hits)

    # add the event id to the processed data frame.
    processed_df.assign(event_id=event_id, inplace=True)

    # Redefine the hit ids and the index to be continuous after the cut hits.
    processed_df.reset_index(drop=True, inplace=True)
    processed_df["hit_id"] = np.arange(len(processed_df))

    # Check if the event has less hits than the minimum required.
    logging.debug(f"Event {event_id} contains {len(processed_df)} hits.")
    if len(processed_df) < min_hits:
        logging.info(
            f"Event {event_id} has only {len(processed_df)} hits! Skipping event..."
        )
        return np.array(
            [
                event_id,
                0,
                0,
                0,
                0,
                len(processed_df),
                n_zero_charge,
                n_multi_hits,
            ],
            dtype=int,
        )

    # Get the true edges using the true time order of the hits
    true_edges = get_time_ordered_true_edges(processed_df)
    logging.info(
        f"Time ordered truth graph built for {event_id} with size {true_edges.shape}"
    )

    # Build input edges by connecting all hits to all other hits.
    if input_edge_method == "all":
        input_edges = get_all_edges(processed_df)
    elif input_edge_method == "layerwise":
        input_edges = get_layerwise_graph(
            processed_df,
            filtering=kwargs["filtering"],
            inneredges=kwargs["inneredges"],
            directional=kwargs["directional"],
        )
    elif input_edge_method == "layerwise_v2":
        input_edges = get_layerwise_graph_v2(processed_df, kwargs["filtering"])

    logging.info(
        f"Input graph built with method {input_edge_method} for {event_id} with size {input_edges.shape}"
    )

    # feature scale for X=[r,phi,isochrone] (basically a normalization for the input features)
    feature_scale = [42, np.pi, 0.5]

    # Build the PyTorch Geometric (PyG) 'Data' object
    data = Data(
        x=torch.from_numpy(
            processed_df[["r", "phi", "isochrone"]].to_numpy() / feature_scale
        ).float(),
        pid=torch.from_numpy(processed_df["particle_id"].to_numpy()),
        hid=torch.from_numpy(processed_df["hit_id"].to_numpy()),
        pt=torch.from_numpy(processed_df["ppt"].to_numpy()),
        vertex=torch.from_numpy(processed_df[["vx", "vy", "vz"]].to_numpy()),
        pdgcode=torch.from_numpy(processed_df["pdgcode"].to_numpy()),
        ptheta=torch.from_numpy(processed_df["ptheta"].to_numpy()),
        peta=torch.from_numpy(processed_df["peta"].to_numpy()),
        pphi=torch.from_numpy(processed_df["pphi"].to_numpy()),
        true_edges=torch.from_numpy(true_edges),
        primary=torch.from_numpy(processed_df["primary"].to_numpy()),
        dep_charge=torch.from_numpy(processed_df["dep_charge"].to_numpy()),
        r_in=torch.from_numpy(processed_df["r_in"].to_numpy()),
        r_out=torch.from_numpy(processed_df["r_out"].to_numpy()),
        event_file=event_id,
    )

    # Get the input and true edges as PyTorch tensors
    input_edges = torch.from_numpy(input_edges)
    true_edges = data.true_edges

    # Divide the number of true edges by 2 to get the "real" number of true edges because they are bidirectional
    logging.info(f"Number of true edges: {int(true_edges.shape[1] / 2)}")

    # Label the input edges, and reorganizes the order of the edges to fit the labels
    new_input_edges, y = graph_intersection(input_edges, true_edges)

    # Save both the labels and edges in the data object
    data.edge_index = new_input_edges
    data.y_pid = y

    logging.info(f"Number of input edges: {new_input_edges.shape[1]}")
    logging.info(f"Number of true input edges: {y[y==True].shape[0]}")
    logging.info(f"Number of false input edges: {y[y==False].shape[0]}")

    # Save the data object to a PyTorch file
    with open(output_filename, "wb") as output_file:
        torch.save(data, output_file)

    # Update the progress bar
    progress_bar.update(n=1)

    # Return some meta information of the event
    return np.array(
        [
            event_id,
            int(true_edges.shape[1] / 2),
            new_input_edges.shape[1],
            y[y == True].shape[0],
            y[y == False].shape[0],
            len(processed_df),
            n_zero_charge,
            n_multi_hits,
        ],
        dtype=int,
    )
