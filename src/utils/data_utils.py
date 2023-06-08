# Adapted from: https://github.com/seedatnabeel/TE-CDE/blob/main/src/utils/data_utils.py

import pickle
import random
import numpy as np
import torch
import torchcde
from scipy.stats import boxcox


def get_processed_data(raw_sim_data, scaling_params):
    """
    It takes the raw simulation data and the scaling parameters, and returns a dictionary with the
    following keys:

    - `current_covariates`: the current covariates (cancer volume and patient type)
    - `time_covariates`: the time covariates (intensity)
    - `previous_treatments`: the previous treatments (one-hot encoded)
    - `current_treatments`: the current treatments (one-hot encoded)
    - `outputs`: the outputs (cancer volume)
    - `active_entries`: the active entries (1 if the patient is still alive, 0 otherwise)
    - `unscaled_outputs`: the unscaled outputs (cancer volume)
    - `input_means`: the input means (cancer volume, patient type, chemo application, radio application)
    - `inputs_stds`: the input standard deviations (cancer volume,

    raw_sim_data (dict): the dataframe containing the simulation data
    scaling_params (tuple): the mean and standard deviation of the cancer volume and patient types

    CODE ADAPTED FROM: https://github.com/ioanabica/Counterfactual-Recurrent-Network
    """
    mean, std = scaling_params

    max_horizon = 5

    mean["chemo_application"] = 0
    mean["radio_application"] = 0
    std["chemo_application"] = 1
    std["radio_application"] = 1

    input_means = mean[
        ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
    ].values.flatten()
    input_stds = std[
        ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
    ].values.flatten()

    # Normalize:
    cancer_volume = (raw_sim_data["cancer_volume"] - mean["cancer_volume"]) / std[
        "cancer_volume"
    ]
    patient_types = (raw_sim_data["patient_types"] - mean["patient_types"]) / std[
        "patient_types"
    ]
    patient_arms = raw_sim_data["patient_arms"]
    data_x_intensity = raw_sim_data["intensity_covariates"]

    patient_types = np.stack(
        [patient_types for t in range(cancer_volume.shape[1])],
        axis=1,
    )
    data_x_intensity = np.stack(
        [data_x_intensity for t in range(cancer_volume.shape[1])],
        axis=1,
    )

    patient_arms = np.stack(
        [patient_arms for t in range(cancer_volume.shape[1])],
        axis=1,
    )

    # Continuous values
    intensity = raw_sim_data["intensity"]

    # Binary application
    chemo_application = raw_sim_data["chemo_application"]
    radio_application = raw_sim_data["radio_application"]
    sequence_lengths = raw_sim_data["sequence_lengths"]

    # Convert treatments to one-hot encoding
    treatments = np.concatenate(
        [
            chemo_application[:, :, np.newaxis],
            radio_application[:, :, np.newaxis],
        ],
        axis=-1,
    )

    one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 4))
    for patient_id in range(treatments.shape[0]):
        for timestep in range(treatments.shape[1]):
            if (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 0
            ):
                one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 0
            ):
                # time_since_treat_zero = 0
                one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 1
            ):
                # time_since_treat_one = 0
                one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 1
            ):
                # time_since_treat_both = 0
                one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]
            elif (  # Not observed, i.e., no treatment applied (all treatments are assumed to be observed)
                np.isnan(treatments[patient_id][timestep][0])
                or np.isnan(treatments[patient_id][timestep][1])
            ):
                one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]

    # Add treatments for entire duration
    one_hot_treatments_current = one_hot_treatments[:, :-max_horizon, :]
    one_hot_treatments_future = one_hot_treatments[:, :, :]

    # Current covariates are the patient's current cancer volume and type; optionally also the intensity covariates
    if data_x_intensity.shape[-1] == 0:  # Do not add intensity covariates (as there are none)
        current_covariates = np.concatenate([cancer_volume[:, :-max_horizon, np.newaxis],  # current output
                                             patient_types[:, :-max_horizon, np.newaxis],
                                             ],
                                            axis=-1)
    else:
        current_covariates = np.concatenate([cancer_volume[:, :-max_horizon, np.newaxis],  # current output
                                             patient_types[:, :-max_horizon, np.newaxis],
                                             data_x_intensity[:, :-max_horizon, :],
                                             ],
                                            axis=-1)

    # Ground truth observation probabilities:
    obs_probabilities = raw_sim_data["obs_probabilities"]
    obs_prob_one_step = obs_probabilities[:, 1:-max_horizon + 1, np.newaxis]
    obs_prob_two_step = obs_probabilities[:, 2:-max_horizon + 2, np.newaxis]
    obs_prob_three_step = obs_probabilities[:, 3:-max_horizon + 3, np.newaxis]
    obs_prob_four_step = obs_probabilities[:, 4:-max_horizon + 4, np.newaxis]
    obs_prob_five_step = obs_probabilities[:, 5:, np.newaxis]
    obs_probabilities = np.concatenate((obs_prob_one_step,
                                        obs_prob_two_step,
                                        obs_prob_three_step,
                                        obs_prob_four_step,
                                        obs_prob_five_step), axis=-1)

    # time_covariates = np.concatenate([intensity[:, 0:, np.newaxis]], axis=-1)
    outcome_one_step = cancer_volume[:, 1:-max_horizon + 1, np.newaxis]
    outcome_two_step = cancer_volume[:, 2:-max_horizon + 2, np.newaxis]
    outcome_three_step = cancer_volume[:, 3:-max_horizon + 3, np.newaxis]
    outcome_four_step = cancer_volume[:, 4:-max_horizon + 4, np.newaxis]
    outcome_five_step = cancer_volume[:, 5:, np.newaxis]
    outputs = np.concatenate((outcome_one_step,
                              outcome_two_step,
                              outcome_three_step,
                              outcome_four_step,
                              outcome_five_step), axis=-1)

    output_means = mean[["cancer_volume"]].values.flatten()[0]  # because we only need scalars here
    output_stds = std[["cancer_volume"]].values.flatten()[0]

    # Add active entires (observed time steps)
    active_entries = (~np.isnan(outputs)).astype(int)

    raw_sim_data["current_covariates"] = current_covariates
    # raw_sim_data["time_covariates"] = time_covariates
    # raw_sim_data["previous_treatments"] = one_hot_previous_treatments
    raw_sim_data["current_treatments"] = one_hot_treatments_current
    raw_sim_data["future_treatments"] = one_hot_treatments_future
    raw_sim_data["outputs"] = outputs
    raw_sim_data["active_entries"] = active_entries

    raw_sim_data["obs_probabilities"] = obs_probabilities

    raw_sim_data["input_means"] = input_means
    raw_sim_data["inputs_stds"] = input_stds
    raw_sim_data["output_means"] = output_means
    raw_sim_data["output_stds"] = output_stds

    return raw_sim_data


def process_data(pickle_map):
    """
    Returns processed train, val, test data from pickle map

    Args:
    pickle_map (dict): dict containing data from pickle map

    Returns:
    training_processed (np array): training data processed numpy
    validation_processed (np array): validation data processed numpy
    test_processed (np array): test data processed numpy
    """
    # load data from pickle_map
    training_data = pickle_map["training_data"]
    validation_data = pickle_map["validation_data"]
    test_cf_data = pickle_map["test_data"]     # Counterfactuals
    test_f_data = pickle_map["test_data_factuals"]      # Factuals
    scaling_data = pickle_map["scaling_data"]

    # get processed data
    training_processed = get_processed_data(training_data, scaling_data)
    validation_processed = get_processed_data(validation_data, scaling_data)
    test_f_processed = get_processed_data(test_f_data, scaling_data)
    test_cf_processed = get_processed_data(test_cf_data, scaling_data)

    return training_processed, validation_processed, test_f_processed, test_cf_processed


def data_to_torch_tensor(data, sample_prop=1, time_concat=-1):
    """
    Returns torch tensors of data -- one step ahead

    Args:
    data (numpy array): np array containing data
    sample_prop (int): proportion of samples

    Returns:
    data_X (torch tensor): containing covariates
    data_A (torch tensor): containing previous_treatments
    data_y (torch tensor): containing outcomes
    data_tr (torch tensor): containing current treatments
    """

    # extract data
    data_x = data["current_covariates"]
    data_tr = data["current_treatments"]
    data_tr_tau = data["future_treatments"]
    data_y = data["outputs"]
    data_ae = data["active_entries"]

    data_time = None  # Because we include time in data_x

    obs_prob = data["obs_probabilities"]

    total_samples = data_x.shape[0]

    # get samples based on sampling proportion
    sample_prop = 1
    samples = int(total_samples * sample_prop)

    # numpy to torch tensor
    sample_idxs = random.sample(range(0, total_samples), samples)

    data_X = torch.from_numpy(data_x[:, :, :])
    data_y = torch.from_numpy(data_y[:, :, :])
    data_ae = torch.from_numpy(data_ae)
    data_tr = torch.from_numpy(data_tr[:, :, :])
    data_tr_tau = torch.from_numpy(data_tr_tau[:, :, :])
    data_obs_prob = torch.from_numpy(obs_prob[:, :])

    return (
        data_X,
        data_tr,
        data_y,
        data_ae,
        data_tr_tau,
        data_obs_prob
    )


def write_to_file(contents, filename):
    """
    It takes in a variable called contents and a variable called filename, and
    then writes the contents to a pickle file with the name filename.

    contents (str): the data to be written to the file
    filename (str): the name of the file to write to
    """
    # write contents to pickle file

    with open(filename, "wb") as handle:
        pickle.dump(contents, handle)


def read_from_file(filename):
    """
    It loads the file from pickle.

    filename (str): the name of the file to read from
    return: A list of dictionaries.
    """
    # load file from pickle

    return pickle.load(open(filename, "rb"))
