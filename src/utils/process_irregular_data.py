# Adapted from: https://github.com/seedatnabeel/TE-CDE/blob/main/src/utils/process_irregular_data.py

import logging
import math
import random
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import Hawkes as hk


def simulate_hawkes(mu, alpha, beta, start, end):
    """
    It simulates a Hawkes process with the given parameters and returns the event times

    mu (float): the baseline intensity
    alpha (float): the intensity of the process
    beta (float): the decay rate of the exponential kernel
    start (int): the start time of the simulation
    end (int): the end time of the simulation
    return: a list of times at which events occur.
    """
    model = hk.simulator()
    model.set_kernel("exp")
    model.set_baseline("const")

    para = {"mu": mu, "alpha": alpha, "beta": beta}
    model.set_parameter(para)

    itv = [start, end]  # the observation interval
    T = model.simulate(itv)

    T = np.unique([int(x) for x in T])

    return T


def get_timestamps(states, kappa, process="Hawkes", intensities=None):
    """
    The function takes in a list of states and a kappa value, and returns a list of timestamps that
    are generated using a Hawkes process

    states (list): a list of integers, where each integer represents a state
    kappa (float): the ratio of the intensity of the next state to the current state
    return: A list of timestamps
    """
    if process == "Hawkes":
        # state indices
        s1_idx = np.where(np.array(states) == 1)[0]
        s2_idx = np.where(np.array(states) == 2)[0]
        s3_idx = np.where(np.array(states) == 3)[0]
        s4_idx = np.where(np.array(states) == 4)[0]

        # apply kappa
        mu_s1 = 0.01
        mu_s2 = kappa * mu_s1
        mu_s3 = kappa * mu_s2
        mu_s4 = kappa * mu_s3

        # simulate Hawkes per cancer state
        overall_timestamps = []
        if len(s4_idx) > 1:
            s4_timestamps = simulate_hawkes(
                mu=mu_s4,
                alpha=1,
                beta=2,
                start=s4_idx[0],
                end=s4_idx[-1],
            )
            overall_timestamps.extend(s4_timestamps)

        if len(s3_idx) > 1:
            s3_timestamps = simulate_hawkes(
                mu=mu_s3,
                alpha=0.5,
                beta=2,
                start=s3_idx[0],
                end=s3_idx[-1],
            )
            overall_timestamps.extend(s3_timestamps)

        if len(s2_idx) > 1:
            s2_timestamps = simulate_hawkes(
                mu=mu_s2,
                alpha=0.1,
                beta=2,
                start=s2_idx[0],
                end=s2_idx[-1],
            )
            overall_timestamps.extend(s2_timestamps)

        if len(s1_idx) > 1:
            s1_timestamps = simulate_hawkes(
                mu=mu_s1,
                alpha=0.01,
                beta=2,
                start=s1_idx[0],
                end=s1_idx[-1],
            )
            overall_timestamps.extend(s1_timestamps)
    elif process == "Poisson":
        overall_timestamps = np.flatnonzero(np.random.binomial(1, intensities))

    return overall_timestamps


def sim_hawkes(states, kappa, intensities=None):
    """
    > This function simulates a Hawkes process with the given states and kappa

    states (list): a list of lists, where each list is a list of timestamps for a particular state
    kappa (float): the intensity of the hawkes process
    return: The timestamps of the events.
    """

    done = False
    i = 0
    while done == False:
        overall_timestamps = get_timestamps(states, kappa, process="Poisson", intensities=intensities)

        t_len = len(overall_timestamps)
        state_len = len(states)

        done = True

        i += 1
        if i > 10:
            break

    return overall_timestamps


def convert_to_cancer_stage(row):
    """
    > This function takes a list of numbers, converts them to a diameter in centimeters, and then returns a list of
    cancer stages

    row: the row of the dataframe that we're currently on
    return: A list of the cancer stages for each tumor.
    """

    stage_list = []

    for idx, number in enumerate(row):

        diameter_cm = (number / (math.pi / 6)) ** (1.0 / 3.0)

        if diameter_cm < 3:
            stage = 1
        elif diameter_cm >= 3 and diameter_cm < 4:
            stage = 2
        elif diameter_cm >= 4 and diameter_cm < 5:
            stage = 3
        elif diameter_cm >= 5:
            stage = 4
        elif np.isnan(diameter_cm):
            stage = np.nan

        stage_list.append(stage)

    return stage_list


def masked_vector(input_vector, mask_indices, mask_type=0):
    """
    > This function takes a vector, a list of indices, and a mask type, and returns a vector with the values at the
    indices replaced by the mask type

    input_vector: The vector to be masked
    mask_indices: The indices of the input vector that we want to mask
    mask_type: 0 or 1. 0 means that the masked indices will be set to 0, 1 means that the masked
    indices will be set to 1, defaults to 0 (optional)
    A list of values from the input_vector, where the values at the indices specified in
    mask_indices are replaced with the value specified in mask_type.
    """

    mask_vector = [
        val if idx in mask_indices else mask_type
        for idx, val in enumerate(input_vector)
    ]

    return mask_vector


def get_intensity_vector(vector):
    """
    > This function takes a vector of 0s and 1s and returns a vector of the same length where each element is the
    number of 1s that have been seen so far - representing the intensity of sampling

    vector: the vector of values to be converted
    return: the intensity vector.
    """
    count = 0

    intensity = []

    for val in vector:
        if np.isnan(val):
            intensity.append(count)
        else:
            count += 1
            intensity.append(count)

    return intensity


def data_sampler(
    data,
    data_split,
    interpolate=False,
    strategy="all",
    sample_prop=1,
    kappa=10,
    max_samples=5000,
):

    """
    > This function takes in a data dictionary, a data split, and a few other parameters, and returns a
    data dictionary with the same keys as the input data dictionary, but with the values sampled
    according to the Hawkes process.

    The function first creates a copy of the data dictionary, and then samples the data according to the
    number of samples specified by the user.

    It then iterates through each sample, and determines whether the sample is treated or
    untreated. If the sample is untreated, the function uses a Hawkes process with a kappa value of 1.
    If the sample is treated, the function uses a Hawkes process with a kappa value of 10.

    It then uses the Hawkes process to generate a list of indices, and then uses these indices
    to mask the data.

    Finally, it then returns the data dictionary with the masked data.

    data: the data dictionary
    data_split: The data split you want to sample from
    interpolate: If True, interpolates the missing values in the data, defaults to False
    (optional)
    strategy: "all" or "random", defaults to all (optional)
    sample_prop: The proportion of the data to sample, defaults to 1 (optional)
    kappa: the intensity of the hawkes process, defaults to 10 (optional)
    max_samples: The maximum number of samples to be taken from the data, defaults to 5000
    (optional)
    """

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    raw_data = deepcopy(data[data_split])

    # Process data:
    cancer_volume = raw_data["cancer_volume"]

    cancer_stages = np.apply_along_axis(convert_to_cancer_stage, 0, cancer_volume)
    features = list(raw_data.keys())

    # For plotting:
    linestyles = ['-', '--', '-.', ':', (0, (1, 1))]
    colors = ['green', 'red', 'black', 'blue', 'orange']
    max_idx = 5

    # Sample per patient
    for idx, indiv_stage in enumerate(cancer_stages):
        if np.sum(raw_data["radio_application"][idx]) <= 2 and np.sum(
            raw_data["chemo_application"][idx] <= 2,
        ):  # Never treated
            untreated = True
            treated = False
            kappa = 1
        else:
            untreated = False
            treated = True
            kappa = 10

        if idx % 100 == 0:
            print(idx, ": Running - Hawkes")

        final_hawkes = []
        final_hawkes.extend(sim_hawkes(indiv_stage, kappa, intensities=raw_data["obs_probabilities"][idx]))
        # Take all unique timesteps:
        final_sampled_indices = np.unique(final_hawkes)

        for feature in features:
            if feature in [
                "cancer_volume",
                "chemo_dosage",
                "radio_dosage",
                # "chemo_application",
                # "radio_application",
                "chemo_probabilities",
                "radio_probabilities",
            ]:
                mask_vec = masked_vector(
                    input_vector=raw_data[feature][idx],
                    mask_indices=final_sampled_indices,
                    # mask_type=0,
                    mask_type=float('nan'),
                )
                if interpolate == True:
                    mask_vec = np.array(mask_vec, dtype=np.float64)
                    nans, x = nan_helper(mask_vec)
                    mask_vec[nans] = np.interp(x(nans), x(~nans), mask_vec[~nans])

                # Plot all values (unobserved)
                # if idx < max_idx:
                #     if feature == "cancer_volume":
                #         # Save these (not overriding them)
                #         unobserved_outcomes_idx = raw_data[feature][idx].copy()
                #         plt.plot(raw_data[feature][idx], linestyle=linestyles[idx % 5], color=colors[idx % 5],
                #                  label="Cancer volume", alpha=0.25)
                #     elif feature == "chemo_application":
                #         plt.plot(np.arange(raw_data[feature].shape[1])[raw_data[feature][idx] > 0],
                #                  unobserved_outcomes_idx[raw_data[feature][idx] > 0],
                #                  marker='x', linestyle='None', color=colors[idx % 5], label="Chemo", alpha=0.4)
                #     elif feature == "radio_application":
                #         plt.plot(np.arange(raw_data[feature].shape[1])[raw_data[feature][idx] > 0],
                #                  unobserved_outcomes_idx[raw_data[feature][idx] > 0],
                #                  marker='+', linestyle='None', color=colors[idx % 5], label="Radio", alpha=0.4)

                # Apply mask:
                raw_data[feature][idx] = mask_vec

               # Plot observations:
    #             if idx < max_idx:
    #                 if feature == "cancer_volume":
    #                     plt.plot(np.arange(raw_data[feature].shape[1])[np.array(mask_vec) > 0],
    #                              unobserved_outcomes_idx[np.array(mask_vec) > 0],
    #                              marker='o', linestyle='None', color=colors[idx % 5], label="Chemo", alpha=0.3)
    #                     plt.plot(np.arange(raw_data[feature].shape[1])[np.array(mask_vec) == 0],
    #                              unobserved_outcomes_idx[np.array(mask_vec) == 0],
    #                              marker='o', linestyle='None', color=colors[idx % 5], mfc='None', label="Chemo", alpha=0.3)
    #                 elif feature == "chemo_application":
    #                     plt.plot(np.arange(raw_data[feature].shape[1])[np.array(mask_vec) > 0],
    #                              unobserved_outcomes_idx[np.array(mask_vec) > 0],
    #                              marker='x', linestyle='None', color=colors[idx % 5], label="Chemo", alpha=1)
    #                 elif feature == "radio_application":
    #                     plt.plot(np.arange(raw_data[feature].shape[1])[np.array(mask_vec) > 0],
    #                              unobserved_outcomes_idx[np.array(mask_vec) > 0],
    #                              marker='+', linestyle='None', color=colors[idx % 5], label="Radio", alpha=1)
    # plt.yscale('log')
    # plt.title("Tumor volume observations over time")
    # plt.gcf().set_dpi(300)
    # mng = plt.get_current_fig_manager()
    # plt.show()

    raw_data["intensity"] = np.apply_along_axis(
        get_intensity_vector,
        1,
        raw_data["cancer_volume"],
    )
    print('Average number of observations:', raw_data["intensity"].max(axis=1).mean())

    return raw_data


def transform_data(
    data,
    interpolate=False,
    strategy="all",
    sample_prop=1,
    kappa=10,
    max_samples=5000,
):
    """
    > This function takes in a dictionary of data, and returns a dictionary of data,
    where the data has been transformed & sampled

    data: the data dictionary
    interpolate: whether to interpolate the data
    strategy: 'all' or 'random', defaults to all (optional)
    sample_prop: the proportion of the data to sample, defaults to 1 (optional)
    kappa: the number of samples to take from each class, defaults to 10 (optional)
    max_samples: the maximum number of samples to be drawn from the data, defaults to 5000
    (optional)
    return: The data is being returned with the updated data.
    """

    logging.info("Transforming data - sampler")

    updated_data = {}

    updated_data["training_data"] = data_sampler(
        data=data,
        data_split="training_data",
        interpolate=interpolate,
        strategy=strategy,
        sample_prop=sample_prop,
        kappa=kappa,
        max_samples=max_samples,
    )
    updated_data["validation_data"] = data_sampler(
        data=data,
        data_split="validation_data",
        interpolate=interpolate,
        strategy=strategy,
        sample_prop=sample_prop,
        kappa=kappa,
        max_samples=max_samples,
    )
    # We do not sample for test data (all timesteps observed, i.e., no informative sampling):
    # Save hypothetical observation_probabilities (for evaluating observation probability/intensity predictions)
    obs_prob_hypothetical_factual = data["test_data_factuals"]["obs_probabilities"]
    obs_prob_hypothetical = data["test_data"]["obs_probabilities"]
    # Overwrite with intensity = one
    data["test_data_factuals"]["obs_probabilities"] = np.ones_like(data["test_data_factuals"]["obs_probabilities"])
    data["test_data"]["obs_probabilities"] = np.ones_like(data["test_data"]["obs_probabilities"])
    # "Sample"
    updated_data['test_data_factuals'] = data_sampler(
        data=data,
        data_split='test_data_factuals',
        interpolate=interpolate,
        strategy=strategy,
        sample_prop=sample_prop,
        kappa=kappa
    )
    updated_data["test_data"] = data_sampler(
        data=data,
        data_split="test_data",
        interpolate=interpolate,
        strategy=strategy,
        sample_prop=sample_prop,
        kappa=kappa,
        max_samples=max_samples,
    )
    # Add obs_probabilities back for evaluation
    updated_data["test_data_factuals"]["obs_probabilities"] = obs_prob_hypothetical_factual
    updated_data["test_data"]["obs_probabilities"] = obs_prob_hypothetical

    updated_data["scaling_data"] = data["scaling_data"]

    return updated_data
