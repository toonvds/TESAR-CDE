"""
CODE ADAPTED FROM: https://github.com/sjblim/rmsn_nips_2018 &
https://github.com/ioanabica/Counterfactual-Recurrent-Network &
https://github.com/seedatnabeel/TE-CDE/blob/main/src/utils/cancer_simulation.py

Medically realistic data simulation for small-cell lung cancer based on Geng et al 2017.
URL: https://www.nature.com/articles/s41598-017-13646-z

Notes:
- Simulation time taken to be in days

"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import (
    truncnorm,  # we need to sample from truncated normal distributions
)
from scipy.special import expit

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Constants

# Spherical calculations - tumours assumed to be spherical per Winer-Muram et al 2002.
# URL: https://pubs.rsna.org/doi/10.1148/radiol.2233011026?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub%3Dpubmed
def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0


def calc_diameter(volume):
    return ((volume / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)) * 2.0


# Tumour constants per
tumour_cell_density = 5.8 * 10.0**8.0  # cells per cm^3
tumour_death_threshold = calc_volume(13)  # assume spherical
# tumour_death_threshold = np.inf   # No deaths

# Patient cancer stage. (mu, sigma, lower bound, upper bound) - for lognormal dist
tumour_size_distributions = {
    "I": (1.72, 4.70, 0.3, 5.0),
    "II": (1.96, 1.63, 0.3, 13.0),
    "IIIA": (1.91, 9.40, 0.3, 13.0),
    "IIIB": (2.76, 6.87, 0.3, 13.0),
    "IV": (3.86, 8.82, 0.3, 13.0),
}  # 13.0 is the death condition

# Observations of stage proportions taken from Detterbeck and Gibson 2008
# - URL: http://www.jto.org/article/S1556-0864(15)33353-0/fulltext#cesec50\
cancer_stage_observations = {
    "I": 1432,
    "II": 128,
    "IIIA": 1306,
    "IIIB": 7248,
    "IV": 12840,
}


EPS = 1e-8


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulation Functions


def get_confounding_params(num_patients, chemo_coeff, radio_coeff, obs_coeff, intensity_cov, num_time_steps):
    """

    Get original simulation parameters, and add extra ones to control confounding

    :param num_patients:
    :param chemo_coeff: Bias on action policy for chemotherapy assignments
    :param radio_activation_group: Bias on action policy for radiotherapy assignments
    :return:
    """

    basic_params = get_standard_params(num_patients)
    patient_types = basic_params["patient_types"]
    tumour_stage_centres = [s for s in cancer_stage_observations if "IIIA" not in s]
    tumour_stage_centres.sort()

    basic_params["num_time_steps"] = num_time_steps

    d_max = calc_diameter(tumour_death_threshold)
    basic_params["chemo_sigmoid_intercepts"] = np.array(
        [d_max / 2.0 for i in patient_types],
    )
    basic_params["radio_sigmoid_intercepts"] = np.array(
        [d_max / 2.0 for i in patient_types],
    )
    basic_params["obs_sigmoid_intercepts"] = np.array(
        [d_max / 2.0 for i in patient_types],
    )

    basic_params["chemo_sigmoid_betas"] = np.array(
        [chemo_coeff / d_max for i in patient_types],
    )
    basic_params["radio_sigmoid_betas"] = np.array(
        [radio_coeff / d_max for i in patient_types],
    )
    basic_params["obs_sigmoid_betas"] = np.array(
        [obs_coeff / d_max for i in patient_types],
    )

    # RCT: randomize patients; Confounding: based on initial stage
    # Either sequential or combined treatment
    # - Sequential treatment: chemo -> radio
    seq_chemo = np.zeros(num_time_steps)
    seq_radio = np.zeros(num_time_steps)
    for i in range(num_time_steps):
        # i + 1 to have no treatment on timestep = 0
        # 5 weeks of weekly chemo
        if (i + 1) % 7 == 0 and i > 21 and i + 1 <= 56:
            seq_chemo[i] = 1.
        # From 6 weeks: weekly radio
        if (i + 1) % 7 == 0 and i > 21 and i + 1 > 56:
            seq_radio[i] = 1.
        if i > 91:
            break
    # - Combined treatment: chemo + radio
    com_chemo = np.zeros(num_time_steps)
    com_radio = np.zeros(num_time_steps)
    for i in range(num_time_steps):
        # Every two weeks chemo and radio
        if (i + 1) % 14 == 0 and i > 21:
            com_chemo[i] = 1.
        if (i + 1) % 14 == 0 and i > 21:
            com_radio[i] = 1.
        if i > 91:
            break

    # Assign treatments:
    # Sequential = 0; Combined = 1
    patient_arms = np.random.binomial(1, 0.5, num_patients)
    basic_params["assigned_actions"] = np.zeros((num_patients, num_time_steps, 2))
    basic_params["assigned_actions"][patient_arms == 0, :, 0] = seq_chemo
    basic_params["assigned_actions"][patient_arms == 0, :, 1] = seq_radio
    basic_params["assigned_actions"][patient_arms == 1, :, 0] = com_chemo
    basic_params["assigned_actions"][patient_arms == 1, :, 1] = com_radio

    # Counterfactual treatments:
    basic_params["counterfactual_actions"] = np.zeros((num_patients, num_time_steps, 2))
    basic_params["counterfactual_actions"][patient_arms == 1, :, 0] = seq_chemo
    basic_params["counterfactual_actions"][patient_arms == 1, :, 1] = seq_radio
    basic_params["counterfactual_actions"][patient_arms == 0, :, 0] = com_chemo
    basic_params["counterfactual_actions"][patient_arms == 0, :, 1] = com_radio

    basic_params["patient_arms"] = patient_arms

    # Get intensity covariates
    basic_params["intensity_covariates"] = np.random.normal(0, 1, (num_patients, intensity_cov))

    return basic_params


def get_standard_params(num_patients):  # additional params
    """
    Simulation parameters from the Nature article + adjustments for static variables

    :param num_patients:
    :return: simulation_parameters
    """

    # Adjustments for static variables
    possible_patient_types = [1, 2, 3]
    patient_types = np.random.choice(possible_patient_types, num_patients)
    chemo_mean_adjustments = np.array([0.0 if i < 3 else 0.1 for i in patient_types])
    radio_mean_adjustments = np.array([0.0 if i > 1 else 0.1 for i in patient_types])

    total = 0
    for k in cancer_stage_observations:
        total += cancer_stage_observations[k]
    cancer_stage_proportions = {
        k: float(cancer_stage_observations[k]) / float(total)
        for k in cancer_stage_observations
    }

    # remove possible entries
    possible_stages = list(tumour_size_distributions.keys())
    possible_stages.sort()

    initial_stages = np.random.choice(
        possible_stages,
        num_patients,
        p=[cancer_stage_proportions[k] for k in possible_stages],
    )

    # Get info on patient stages and initial volumes
    output_initial_diam = []
    patient_sim_stages = []
    for stg in possible_stages:
        count = np.sum((initial_stages == stg) * 1)

        mu, sigma, lower_bound, upper_bound = tumour_size_distributions[stg]

        # Convert lognorm bounds in to standard normal bounds
        lower_bound = (np.log(lower_bound) - mu) / sigma
        upper_bound = (np.log(upper_bound) - mu) / sigma

        logging.info(
            (
                "Simulating initial volumes for stage {} "
                + " with norm params: mu={}, sigma={}, lb={}, ub={}"
            ).format(stg, mu, sigma, lower_bound, upper_bound),
        )

        norm_rvs = truncnorm.rvs(
            lower_bound,
            upper_bound,
            size=count,
        )  # truncated normal for realistic clinical outcome

        initial_volume_by_stage = np.exp((norm_rvs * sigma) + mu)
        output_initial_diam += list(initial_volume_by_stage)
        patient_sim_stages += [stg for i in range(count)]

    # Fixed params
    K = calc_volume(30)  # carrying capacity given in cm, so convert to volume
    alpha_beta_ratio = 10
    alpha_rho_corr = 0.87

    # Distributional parameters for dynamics
    parameter_lower_bound = 0.0
    parameter_upper_bound = np.inf
    rho_params = (7 * 10**-5, 7.23 * 10**-3)
    alpha_params = (0.0398, 0.168)
    beta_c_params = (0.028, 0.0007)

    # Get correlated simulation paramters (alpha, beta, rho) which respects bounds
    alpha_rho_cov = np.array(
        [
            [alpha_params[1] ** 2, alpha_rho_corr * alpha_params[1] * rho_params[1]],
            [alpha_rho_corr * alpha_params[1] * rho_params[1], rho_params[1] ** 2],
        ],
    )

    alpha_rho_mean = np.array([alpha_params[0], rho_params[0]])

    simulated_params = []

    while (
        len(simulated_params) < num_patients
    ):  # Keep on simulating till we get the right number of params

        param_holder = np.random.multivariate_normal(
            alpha_rho_mean,
            alpha_rho_cov,
            size=num_patients,
        )

        for i in range(param_holder.shape[0]):

            # Ensure that all params fulfill conditions
            if (
                param_holder[i, 0] > parameter_lower_bound
                and param_holder[i, 1] > parameter_lower_bound
            ):
                simulated_params.append(param_holder[i, :])

        logging.info(
            "Got correlated params for {} patients".format(len(simulated_params)),
        )

    simulated_params = np.array(simulated_params)[
        :num_patients, :
    ]  # shorten this back to normal
    alpha_adjustments = alpha_params[0] * radio_mean_adjustments
    alpha = simulated_params[:, 0] + alpha_adjustments
    rho = simulated_params[:, 1]
    beta = alpha / alpha_beta_ratio

    # Get the remaining indep params
    logging.info("Simulating beta c parameters")
    beta_c_adjustments = beta_c_params[0] * chemo_mean_adjustments
    beta_c = (
        beta_c_params[0]
        + beta_c_params[1]
        * truncnorm.rvs(
            (parameter_lower_bound - beta_c_params[0]) / beta_c_params[1],
            (parameter_upper_bound - beta_c_params[0]) / beta_c_params[1],
            size=num_patients,
        )
        + beta_c_adjustments
    )

    output_holder = {
        "patient_types": patient_types,
        "initial_stages": np.array(patient_sim_stages),
        "initial_volumes": calc_volume(
            np.array(output_initial_diam),
        ),  # assumed spherical with diam
        "alpha": alpha,
        "rho": rho,
        "beta": beta,
        "beta_c": beta_c,
        "K": np.array([K for i in range(num_patients)]),
    }
    # np.random.exponential(expected_treatment_delay, num_patients),

    # Randomise output params
    logging.info("Randomising outputs")
    idx = [i for i in range(num_patients)]
    np.random.shuffle(idx)

    output_params = {}
    for k in output_holder:
        output_params[k] = output_holder[k][idx]

    return output_params


def simulate(simulation_params, assigned_actions=None, intensity_cov_only=False, max_intensity=1):
    """
    Core routine to generate simulation paths

    :param simulation_params:
    :param num_time_steps:
    :param assigned_actions:
    :return:
    """

    total_num_radio_treatments = 1
    total_num_chemo_treatments = 1

    radio_amt = np.array([2.0 for i in range(total_num_radio_treatments)])  # Gy
    radio_days = np.array([i + 1 for i in range(total_num_radio_treatments)])
    chemo_amt = [5.0 for i in range(total_num_chemo_treatments)]
    chemo_days = [(i + 1) * 7 for i in range(total_num_chemo_treatments)]

    # sort this
    chemo_idx = np.argsort(chemo_days)
    chemo_amt = np.array(chemo_amt)[chemo_idx]
    chemo_days = np.array(chemo_days)[chemo_idx]

    drug_half_life = 1  # one day half life for drugs

    # Unpack simulation parameters
    initial_stages = simulation_params["initial_stages"]
    initial_volumes = simulation_params["initial_volumes"]
    alphas = simulation_params["alpha"]
    rhos = simulation_params["rho"]
    betas = simulation_params["beta"]
    beta_cs = simulation_params["beta_c"]
    Ks = simulation_params["K"]
    patient_types = simulation_params["patient_types"]
    window_size = simulation_params["window_size"]  # controls the lookback of the treatment assignment policy

    # Coefficients for treatment assignment probabilities
    chemo_sigmoid_intercepts = simulation_params["chemo_sigmoid_intercepts"]
    radio_sigmoid_intercepts = simulation_params["radio_sigmoid_intercepts"]
    obs_sigmoid_intercepts = simulation_params["obs_sigmoid_intercepts"]
    chemo_sigmoid_betas = simulation_params["chemo_sigmoid_betas"]
    radio_sigmoid_betas = simulation_params["radio_sigmoid_betas"]
    obs_sigmoid_betas = simulation_params["obs_sigmoid_betas"]
    obs_coeff = simulation_params["obs_coeff"]
    patient_arms = simulation_params["patient_arms"]
    intensity_covariates = simulation_params["intensity_covariates"]
    intensity_coefficients = simulation_params["intensity_coefficients"]

    num_patients = initial_stages.shape[0]
    num_time_steps = simulation_params["num_time_steps"]

    # Commence Simulation
    cancer_volume = np.zeros((num_patients, num_time_steps))
    chemo_dosage = np.zeros((num_patients, num_time_steps))
    radio_dosage = np.zeros((num_patients, num_time_steps))
    chemo_application_point = np.zeros((num_patients, num_time_steps))
    radio_application_point = np.zeros((num_patients, num_time_steps))
    sequence_lengths = np.zeros(num_patients)
    death_flags = np.zeros((num_patients, num_time_steps))
    recovery_flags = np.zeros((num_patients, num_time_steps))
    chemo_probabilities = np.zeros((num_patients, num_time_steps))
    radio_probabilities = np.zeros((num_patients, num_time_steps))
    obs_probabilities = np.zeros((num_patients, num_time_steps))

    # Get initial diameters
    cancer_diameters_initial = calc_diameter(initial_volumes)

    # 5% cell variability
    # noise_terms = 0.01 * np.random.randn(num_patients, num_time_steps)
    noise_terms = np.random.normal(0, 0.01, (num_patients, num_time_steps))
    recovery_rvs = np.random.rand(num_patients, num_time_steps)
    # recovery_rvs = np.ones((num_patients, num_time_steps))

    chemo_application_rvs = np.random.rand(num_patients, num_time_steps)
    radio_application_rvs = np.random.rand(num_patients, num_time_steps)

    # Run actual simulation
    for i in range(num_patients):
        if i % 200 == 0:
            logging.info("Simulating patient {} of {}".format(i, num_patients))
        noise = noise_terms[i]

        # initial values
        cancer_volume[i, 0] = initial_volumes[i]
        alpha = alphas[i]
        beta = betas[i]
        beta_c = beta_cs[i]
        rho = rhos[i]
        K = Ks[i]

        # Setup cell volume
        b_death = False
        b_recover = False
        for t in range(0, num_time_steps - 1):  # "- 1" as outcomes are always of the next step

            current_chemo_dose = 0.0
            previous_chemo_dose = 0.0 if t == 0 else chemo_dosage[i, t - 1]

            # Action probabilities + death or recovery simulations
            cancer_volume_used = cancer_volume[i, max(t - window_size, 0): t + 1]
            cancer_diameter_used = np.array(
                [calc_diameter(vol) for vol in cancer_volume_used],
            ).mean()  # mean diameter over 15 days
            cancer_metric_used = cancer_diameter_used

            # probabilities
            if assigned_actions is not None:
                chemo_prob = assigned_actions[i, t, 0]
                radio_prob = assigned_actions[i, t, 1]
            else:
                radio_prob = 1.0 / (
                    1.0
                    + np.exp(
                        -radio_sigmoid_betas[i]
                        * (cancer_metric_used - radio_sigmoid_intercepts[i]),
                    )
                )
                chemo_prob = 1.0 / (
                    1.0
                    + np.exp(
                        -chemo_sigmoid_betas[i]
                        * (cancer_metric_used - chemo_sigmoid_intercepts[i]),
                    )
                )
                # Decrease probabilities - one expected treatment per number of weeks:
                chemo_prob = chemo_prob / 7
                radio_prob = radio_prob / 7

            chemo_probabilities[i, t] = chemo_prob
            radio_probabilities[i, t] = radio_prob

            # Observation probability
            if intensity_cov_only:
                obs_prob = expit(obs_coeff * (
                        np.sum(intensity_coefficients * intensity_covariates[i, :])  # 0 when no covariates included
                        ))
            else:
                obs_prob = max_intensity / (1.0 + np.exp(
                    -obs_coeff * (obs_sigmoid_betas[i] * (cancer_metric_used - obs_sigmoid_intercepts[i]))))

            obs_probabilities[i, t] = obs_prob

            # Action application
            if radio_application_rvs[i, t] < radio_prob:
                # Apply radio treatment
                radio_application_point[i, t] = 1
                radio_dosage[i, t] = radio_amt[0]

            if chemo_application_rvs[i, t] < chemo_prob:
                # Apply chemo treatment
                chemo_application_point[i, t] = 1
                current_chemo_dose = chemo_amt[0]

            # Update chemo dosage
            chemo_dosage[i, t] = (
                previous_chemo_dose * np.exp(-np.log(2) / drug_half_life)
                + current_chemo_dose
            )

            # Model cancer volume evolution
            cancer_volume[i, t + 1] = cancer_volume[i, t] * (
                1
                + rho * np.log(K / cancer_volume[i, t])
                - beta_c * chemo_dosage[i, t]
                - (alpha * radio_dosage[i, t] + beta * radio_dosage[i, t] ** 2)
                + noise[t]
            )  # add noise to fit residuals

            # Censoring:
            # None for now - breaks commented
            # if cancer_volume[i, t + 1] > tumour_death_threshold:
            if cancer_volume[i, t + 1] > np.inf:
                cancer_volume[i, t + 1] = tumour_death_threshold

            # recovery threshold as defined by the previous stuff
            if recovery_rvs[i, t + 1] < np.exp(
                -cancer_volume[i, t + 1] * tumour_cell_density,
            ):
                cancer_volume[i, t + 1] = EPS

        # Package outputs
        sequence_lengths[i] = int(t + 1)

    # Plot some outputs:
    matplotlib.rcParams.update({'font.size': 16})
    plt.style.use('science')
    linestyles = ['-', '--', '-.', ':', (0, (1, 1))]
    markers = ['.', 'o', '+', 'x', '*']
    colors = ['green', 'red', 'black', 'blue', 'orange']

    # plt.figure(figsize=(5, 3))
    # for i in range(min(8, cancer_volume.shape[0])):
    #     plt.plot(cancer_volume[i, :], linestyle=linestyles[i % 5], color=colors[i % 5], label="Cancer volume", alpha=0.8)
    #     # Chemo:
    #     plt.plot(np.arange(num_time_steps)[chemo_application_point[i, :] == 1],
    #              cancer_volume[i, :][chemo_application_point[i, :] == 1],
    #              marker='x', linestyle='None', color=colors[i % 5], label="Chemo", alpha=0.6)
    #     # Radio:
    #     plt.plot(np.arange(num_time_steps)[radio_application_point[i, :] == 1],
    #              cancer_volume[i, :][radio_application_point[i, :] == 1],
    #              marker='+', linestyle='None', color=colors[i % 5], label="Radio", alpha=0.6)
    # plt.yscale('log')
    # plt.ylabel('Tumor volume $Y(t)$')
    # plt.xlabel('Time $t$ (days)')
    # # plt.title("Tumor volume evolution over time")
    # plt.gcf().set_dpi(300)
    # plt.show()
    #
    # # Show some intensities
    # plt.figure(figsize=(5, 3))
    # for i in range(min(8, cancer_volume.shape[0])):
    #     plt.plot(obs_probabilities[i, :-1], linestyle=linestyles[i % 5], color=colors[i % 5], marker=".", alpha=0.2)
    # plt.ylabel('Tumor volume $\lambda(t)$')
    # plt.xlabel('Time $t$ (days)')
    # plt.yscale('log')
    # plt.gcf().set_dpi(300)
    # plt.show()
    #
    # plt.figure(figsize=(4, 4))
    # sns.kdeplot(obs_probabilities.flatten(), fill=0.1, cut=0)
    # # plt.title("$\gamma = " + str(obs_coeff) + "$")
    # plt.gcf().set_dpi(300)
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(figsize=(4, 4))
    # sns.kdeplot(obs_probabilities.mean(axis=1) * obs_probabilities.shape[1], fill=0.1, cut=0)
    # # plt.title("$\gamma = " + str(obs_coeff) + "$")
    # plt.gcf().set_dpi(300)
    # plt.tight_layout()
    # plt.show()

    # for i in range(min(5, cancer_volume.shape[0])):
    #     plt.plot(chemo_dosage[i, :], linestyle=linestyles[i % 5], color=colors[i % 5])
    # plt.title('Chemo dosage')
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()
    # plt.show()
    #
    # for i in range(min(5, cancer_volume.shape[0])):
    #     plt.plot(radio_dosage[i, :], marker=markers[i % 5], color=colors[i % 5], linestyle='None')
    # plt.title('Radio dosage')
    # plt.show()

    outputs = {
        "cancer_volume": cancer_volume,
        "chemo_dosage": chemo_dosage,
        "radio_dosage": radio_dosage,
        "chemo_application": chemo_application_point,
        "radio_application": radio_application_point,
        "chemo_probabilities": chemo_probabilities,
        "radio_probabilities": radio_probabilities,
        "obs_probabilities": obs_probabilities,
        "sequence_lengths": sequence_lengths,
        "patient_types": patient_types,
        "patient_arms": patient_arms,
        "intensity_covariates": intensity_covariates,
    }

    return outputs


def get_scaling_params(sim):
    real_idx = ["cancer_volume", "chemo_dosage", "radio_dosage", "chemo_application", "radio_application"]

    means = {}
    stds = {}
    seq_lengths = sim["sequence_lengths"]
    for k in real_idx:
        active_values = []
        for i in range(seq_lengths.shape[0]):
            end = int(seq_lengths[i])
            active_values += list(sim[k][i, :end])

        means[k] = np.mean(active_values)
        stds[k] = np.std(active_values)

    # Add means for static variables`
    means["patient_types"] = np.mean(sim["patient_types"])
    stds["patient_types"] = np.std(sim["patient_types"])

    return pd.Series(means), pd.Series(stds)


def get_cancer_sim_data(
    chemo_coeff,
    radio_coeff,
    obs_coeff,
    intensity_cov,
    intensity_cov_only,
    max_intensity,
    b_load,
    b_save=False,
    num_patients=10000,
    model_root="results",
    window_size=15,
):
    if window_size == 15:
        pickle_file = os.path.join(
            model_root,
            "new_cancer_sim_{}_{}.p".format(chemo_coeff, radio_coeff),
        )
    else:
        pickle_file = os.path.join(
            model_root,
            "new_cancer_sim_{}_{}_{}.p".format(chemo_coeff, radio_coeff, window_size),
        )

    def _generate():
        num_time_steps = 120

        print('Generating for time steps: ', num_time_steps)

        # Coefficients for the intensity covariates
        intensity_coefficients = np.random.uniform(-1, 1, intensity_cov)

        # Train set
        params = get_confounding_params(
            num_patients=num_patients,
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            obs_coeff=obs_coeff,
            intensity_cov=intensity_cov,
            num_time_steps=num_time_steps,
        )
        params["window_size"] = window_size
        params["obs_coeff"] = obs_coeff
        params["intensity_coefficients"] = intensity_coefficients
        training_data = simulate(params, assigned_actions=params["assigned_actions"],
                                 intensity_cov_only=intensity_cov_only, max_intensity=max_intensity)

        # Validation set
        params = get_confounding_params(
            num_patients=int(num_patients / 4),
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            obs_coeff=obs_coeff,
            intensity_cov=intensity_cov,
            num_time_steps=num_time_steps,
        )
        params["window_size"] = window_size
        params["obs_coeff"] = obs_coeff
        params["intensity_coefficients"] = intensity_coefficients
        validation_data = simulate(params, assigned_actions=params["assigned_actions"],
                                   intensity_cov_only=intensity_cov_only, max_intensity=max_intensity)

        # Test set
        params = get_confounding_params(
            num_patients=num_patients,
            chemo_coeff=chemo_coeff,
            radio_coeff=radio_coeff,
            obs_coeff=obs_coeff,
            intensity_cov=intensity_cov,
            num_time_steps=num_time_steps,
        )
        params["window_size"] = window_size
        params["obs_coeff"] = obs_coeff
        params["intensity_coefficients"] = intensity_coefficients
        test_data_factuals = simulate(params, assigned_actions=params["assigned_actions"],
                                      intensity_cov_only=intensity_cov_only, max_intensity=max_intensity)
        # Only one counterfactual, so we can use standard simulate with the SAME params but COUNTERFACTUAL actions
        test_data_counterfactuals = simulate(params, assigned_actions=params["counterfactual_actions"],
                                             intensity_cov_only=intensity_cov_only, max_intensity=max_intensity)

        scaling_data = get_scaling_params(training_data)

        pickle_map = {
            "chemo_coeff": chemo_coeff,
            "radio_coeff": radio_coeff,
            "num_time_steps": num_time_steps,
            "training_data": training_data,
            "validation_data": validation_data,
            "test_data": test_data_counterfactuals,
            "test_data_factuals": test_data_factuals,
            # "test_data_seq": test_data_seq,
            "scaling_data": scaling_data,
            "window_size": window_size,
        }

        if b_save:
            logging.info("Saving pickle map to {}".format(pickle_file))
            pickle.dump(pickle_map, open(pickle_file, "wb"))
        return pickle_map

    # Controls whether to regenerate the data, or load from a persisted file
    if not b_load:
        pickle_map = _generate()

    else:
        logging.info("Loading pickle map from {}".format(pickle_file))

        try:
            pickle_map = pickle.load(open(pickle_file, "rb"))

        except IOError:
            logging.info(
                "Pickle file does not exist, regenerating: {}".format(pickle_file),
            )
            pickle_map = _generate()

    return pickle_map
