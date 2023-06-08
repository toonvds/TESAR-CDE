import argparse
import logging
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchcde
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.CDE_model import NeuralCDE
from src.utils.data_utils import data_to_torch_tensor
from src.utils.losses import mse
from src.utils.training_tools import EarlyStopping


class trainer:
    def __init__(
        self,
        run,
        hidden_channels_x,
        output_channels,
        hidden_channels_enc,
        hidden_layers_enc,
        hidden_channels_dec,
        hidden_layers_dec,
        hidden_channels_map,
        hidden_layers_map,
        alpha,
        cutoff,
        sample_proportion=1,
        window=7,
        use_time=True,
        importance_weighting=False,
        ground_truth_iw=True,
        multitask=False,
    ):

        self.run = run
        self.model = None
        self.multistep_model = None

        self.device = None

        self.use_time = use_time
        self.hidden_channels_x = hidden_channels_x
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_layers_enc = hidden_layers_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.hidden_layers_dec = hidden_layers_dec
        self.hidden_channels_map = hidden_channels_map
        self.hidden_layers_map = hidden_layers_map
        self.output_channels = output_channels

        self.alpha = alpha
        self.cutoff = cutoff

        self.sample_proportion = sample_proportion

        if self.use_time == False:
            self.time_concat = 0

        self.window = int(window)

        self.importance_weighting = importance_weighting
        self.ground_truth_iw = ground_truth_iw
        self.multitask = multitask

    def _train(self, train_dataloader, model, optimizers):
        model = model.train()

        optimizer, optimizer_intensity = optimizers

        observation_loss = nn.BCEWithLogitsLoss()

        train_losses_total = []
        train_losses_mse = []
        train_losses_obs = []

        for (
            batch_coeffs_x,
            batch_y,
            batch_treat_tau,
            batch_ae,
            batch_iw,
            _,
        ) in train_dataloader:

            batch_coeffs_x = torch.tensor(
                batch_coeffs_x,
                dtype=torch.float,
                device=self.device,
            )

            outcomes = torch.tensor(
                batch_y,
                dtype=torch.float,
                device=self.device,
            )

            active_entries = torch.tensor(
                batch_ae,
                dtype=torch.float,
                device=self.device,
            )

            treat = torch.tensor(
                batch_treat_tau,
                dtype=torch.float,
                device=self.device,
            )

            obs_prob = torch.tensor(
                batch_iw,
                dtype=torch.float,
                device=self.device,
            )

            if model.prediction == "regression":
                pred_y, pred_obs = model(batch_coeffs_x, treat, self.device)

                if self.importance_weighting and not self.ground_truth_iw and self.multitask:
                    # Use learned weights from other arm -- overwrite obs_prob
                    # Intensity loss:
                    obs_loss = observation_loss(pred_obs, active_entries)

                    (self.alpha * obs_loss).backward(retain_graph=True)

                    # Clip gradients for more stable training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer_intensity.step()
                    optimizer_intensity.zero_grad()

                    # Get weights, stabilize and detach:
                    obs_prob = torch.sigmoid(pred_obs)
                    obs_prob = torch.clip(obs_prob, self.cutoff)
                    obs_prob = obs_prob.detach()

                    # compute norm mse loss for predicted outcomes
                    mse_loss = mse(outcomes, pred_y, active_entries, obs_prob)

                    ((1 - self.alpha) * mse_loss).backward()
                    # mse_loss.backward()

                    # Clip gradients for more stable training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                    optimizer.step()
                    optimizer.zero_grad()

                    batch_loss = (1 - self.alpha) * mse_loss + self.alpha * obs_loss

                    train_losses_obs.append(obs_loss.item())
                    train_losses_mse.append(mse_loss.item())
                    train_losses_total.append(batch_loss.item())
                else:
                    # Use saved obs_prob (either from theoretical, two-stage or 1 in case of no weight)

                    # compute norm mse loss for predicted outcomes
                    mse_loss = mse(outcomes, pred_y, active_entries, obs_prob)

                    batch_loss = mse_loss

                    batch_loss.backward()

                    train_losses_mse.append(mse_loss.item())

                    train_losses_total.append(batch_loss.item())

                    # Clip gradients for more stable training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                    optimizer.step()
                    optimizer.zero_grad()

            elif model.prediction == "classification":
                _, pred_obs, integral = model(batch_coeffs_x, treat, self.device)

                # Cross entropy:
                batch_loss = observation_loss(pred_obs, active_entries[:, :, None])

                batch_loss.backward()

                train_losses_total.append(batch_loss.item())

                # Clip gradients for more stable training
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                optimizer.step()
                optimizer.zero_grad()
            else:
                NotImplementedError("Only regression or classification allowed.")

        return model, train_losses_total, train_losses_mse, train_losses_obs

    def _test(self, test_dataloader, model):
        # eval mode
        model = model.eval()

        observation_loss = nn.BCEWithLogitsLoss()

        test_losses_total = []

        with torch.no_grad():
            for (
                batch_coeffs_x_val,
                batch_y_val,
                batch_val_treat,
                batch_ae_val,
                batch_iw_val,
                _
            ) in test_dataloader:

                batch_coeffs_x_val = torch.tensor(
                    batch_coeffs_x_val,
                    dtype=torch.float,
                    device=self.device,
                )

                outcomes_val = torch.tensor(
                    batch_y_val,
                    dtype=torch.float,
                    device=self.device,
                )

                active_entries_val = torch.tensor(
                    batch_ae_val,
                    dtype=torch.float,
                    device=self.device,
                )

                treat_val = torch.tensor(
                    batch_val_treat,
                    dtype=torch.float,
                    device=self.device,
                )

                obs_prob_val = torch.tensor(
                    batch_iw_val,
                    dtype=torch.float,
                    device=self.device,
                )

                if model.prediction == "regression":
                    pred_y_val, pred_obs_val = model(batch_coeffs_x_val, treat_val, self.device)

                    if self.importance_weighting and not self.ground_truth_iw and self.multitask:
                        obs_loss_val = observation_loss(pred_obs_val, active_entries_val)

                        # Get weights, stabilize and detach:
                        obs_prob_val = torch.sigmoid(pred_obs_val)
                        obs_prob_val = torch.clip(obs_prob_val, self.cutoff)

                        # compute norm mse loss - outcomes loss
                        mse_loss_val = mse(outcomes_val, pred_y_val, active_entries_val, obs_prob_val)

                        batch_loss_val = (1 - self.alpha) * mse_loss_val + self.alpha * obs_loss_val
                        # batch_loss_val = mse_loss_val + obs_loss_val
                    else:
                        batch_loss_val = mse(outcomes_val, pred_y_val, active_entries_val, obs_prob_val)

                    # print(torch.mean((torch.sigmoid(pred_int_val) - active_entries_val)**2))
                elif model.prediction == "classification":
                    _, pred_obs_val, _ = model(batch_coeffs_x_val, treat_val, self.device)

                    batch_loss_val = observation_loss(pred_obs_val, active_entries_val[:, :, None])
                else:
                    NotImplementedError("Only regression or classification allowed.")

                # Normalize by number of instances in batch:
                test_losses_total.append(batch_loss_val.item())

        return model, test_losses_total

    def prepare_dataloader(self, data, batch_size=32, importance_weights=None, return_tensors=False,
                           window=7, observed_only=False, process_weights=True):
        horizon = 5

        data_X, data_tr, data_y, data_ae, data_tr_tau, data_obs_prob = data_to_torch_tensor(
            data,
            sample_prop=self.sample_proportion,
        )   # data_A denotes previous treatment, data_tr is current treatment

        # Add importance weights if in data (ground truth or estimated empirically)
        if importance_weights is None:
            importance_weights = torch.ones_like(data_y[:, :, 0])
        elif not torch.is_tensor(importance_weights):
            importance_weights = torch.from_numpy(importance_weights)

        data_concat = torch.cat((data_X, data_tr), 2)  # X covariates and treatment per timestep

        data_shape = list(data_concat.shape)

        # Split for next time step prediction based on window
        total_X = torch.Tensor(size=(data_shape[0] * (data_shape[1] - window), window, data_shape[2]))
        total_y = torch.Tensor(size=(data_shape[0] * (data_shape[1] - window), horizon))
        total_ae = torch.Tensor(size=(data_shape[0] * (data_shape[1] - window), horizon))
        total_tr = torch.Tensor(size=(data_shape[0] * (data_shape[1] - window), horizon, 4))  # 4 treatments (one-hot)
        if process_weights:
            total_iw = torch.Tensor(size=(data_shape[0] * (data_shape[1] - window), horizon))
        else:
            total_iw = importance_weights
        total_obs_prob = torch.Tensor(size=(data_shape[0] * (data_shape[1] - window), horizon))
        print("Splitting data for step(s) prediction(s) -- Max horizon = 5")
        # For each patient:
        for i in range(data_shape[0]):
            # For each timestep:
            for j in range(data_shape[1]):
                start_step = j - window
                if start_step < 0:
                    continue
                # Take all previous timesteps
                X = torch.clone(data_concat[i, :, :])
                # Only from window and observed steps:
                X = X[start_step:j, :]
                # Outcomes
                y = torch.clone(data_y[i, j, :])
                ae = torch.clone(data_ae[i, j, :])
                # Treatments
                tr = torch.clone(data_tr_tau[i, j:j+horizon, :])
                # Importance weights
                if process_weights:
                    iw = torch.clone(importance_weights[i, j])
                # Observation probability
                obs_prob = torch.clone(data_obs_prob[i, j, :])
                # Add to total tensors
                total_X[i * (data_shape[1] - window) + start_step, :, :] = X
                total_y[i * (data_shape[1] - window) + start_step, :] = y
                total_ae[i * (data_shape[1] - window) + start_step, :] = ae
                total_tr[i * (data_shape[1] - window) + start_step, :, :] = tr
                if process_weights:
                    total_iw[i * (data_shape[1] - window) + start_step, :] = iw
                total_obs_prob[i * (data_shape[1] - window) + start_step, :] = obs_prob

        if observed_only:
            # Consider only observed outcomes + histories:
            observed_outcomes, _ = torch.nonzero((~total_y.isnan()).float(), as_tuple=True)
            total_X = total_X[observed_outcomes]
            total_y = total_y[observed_outcomes]
            total_tr = total_tr[observed_outcomes]
            total_ae = total_ae[observed_outcomes]
            total_iw = total_iw[observed_outcomes]
            total_obs_prob = total_obs_prob[observed_outcomes]

        # Interpolate (faster to do this here: only once before training):
        total_X = torchcde.natural_cubic_coeffs(total_X)
        total_tr = torchcde.natural_cubic_coeffs(total_tr)

        if return_tensors:  # For final validation, testing and predict
            return total_X, total_y, total_tr, total_ae, total_iw, total_obs_prob
        else:   # For training
            dataset = torch.utils.data.TensorDataset(total_X, total_y, total_tr, total_ae, total_iw, total_obs_prob)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
            return dataloader, data_shape

    def fit(self, train_data, validation_data, epochs, patience, batch_size):
        logging.info("Getting training data")

        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"

        device = torch.device(device_type)

        self.device = device

        logging.info(f"Predicting using device: {device_type}")

        #############################
        # First, initialize weights #
        #############################
        if not self.importance_weighting:  # Baseline - All weight = 1
            obs_probabilities_train = np.ones_like(train_data["obs_probabilities"])
            obs_probabilities_val = np.ones_like(validation_data["obs_probabilities"])
        elif self.ground_truth_iw:
            obs_probabilities_train = train_data["obs_probabilities"]
            obs_probabilities_val = validation_data["obs_probabilities"]
            # Clipping (optional, but not required for theoretical)
            # obs_probabilities_train[obs_probabilities_train < self.cutoff] = self.cutoff
            # obs_probabilities_val[obs_probabilities_val < self.cutoff] = self.cutoff
        elif self.multitask:  # Multitask - All weight = 1 as they will be learned later
            obs_probabilities_train = np.ones_like(train_data["obs_probabilities"])
            obs_probabilities_val = np.ones_like(validation_data["obs_probabilities"])
        else:       # Two step
            early_stopping = EarlyStopping(patience=patience, delta=0.0001)

            # create dataloaders
            train_dataloader, data_shape = self.prepare_dataloader(
                data=train_data,
                batch_size=int(batch_size),
                window=self.window,
                observed_only=False,
            )
            val_dataloader, _ = self.prepare_dataloader(
                data=validation_data,
                batch_size=int(batch_size),
                window=self.window,
                observed_only=False,
            )

            logging.info("Instantiating Neural CDE Classifier")

            obs_classifier = NeuralCDE(
                input_channels_x=data_shape[2],
                hidden_channels_x=self.hidden_channels_x,
                hidden_channels_enc=self.hidden_channels_enc,
                hidden_layers_enc=self.hidden_channels_enc,
                hidden_channels_dec=self.hidden_channels_dec,
                hidden_layers_dec=self.hidden_layers_dec,
                hidden_channels_map=self.hidden_channels_map,
                hidden_layers_map=self.hidden_layers_map,
                multitask=self.multitask,
                output_channels=1,
                prediction="classification",
            )

            obs_classifier = obs_classifier.to(self.device)

            learning_rate = 5e-4
            optimizer_classifier = torch.optim.Adam(obs_classifier.parameters(), lr=learning_rate, weight_decay=1e-3)
            print('Learning rate classifier:', optimizer_classifier.param_groups[0]['lr'])

            self.run.watch(obs_classifier, log="all")

            logging.info("Training CDE Classifier")
            epochs = epochs

            for epoch in tqdm(range(epochs)):

                logging.info(f"Training epoch: {epoch}")
                # put into train mode
                obs_classifier, train_losses_obs, _, _ = self._train(
                    train_dataloader=train_dataloader,
                    model=obs_classifier,
                    optimizers=[optimizer_classifier, None],
                )

                # zero grad check out + early stopping
                logging.info(f"Validation epoch: {epoch}")
                obs_classifier, val_losses_obs = self._test(
                    test_dataloader=val_dataloader,
                    model=obs_classifier
                )

                tqdm.write(
                    f"Epoch observations: {epoch}: "
                    f"Training loss observations: {np.average(train_losses_obs)}; "
                    f"Val loss observations: {np.average(val_losses_obs)}")

                if np.average(train_losses_obs,) == float("nan"):
                    raise ValueError("Exiting run...")

                self.run.log(
                    {
                        "Epoch observations": epoch,
                        "Training loss observations": np.average(train_losses_obs),
                        "Val loss observations": np.average(val_losses_obs),
                    },
                )

                early_stopping(np.average(val_losses_obs), obs_classifier)

                if early_stopping.early_stop:
                    print("Early stopping phase initiated...")
                    break

            # Use predicted importance weights:
            x_train = torch.tensor(
                train_dataloader.dataset.tensors[0],
                dtype=torch.float,
                device=self.device,
            )
            treat_train = torch.tensor(
                train_dataloader.dataset.tensors[2],
                dtype=torch.float,
                device=self.device,
            )
            obs_probabilities_train, _, _ = obs_classifier(x_train, treat_train, self.device)

            x_val = torch.tensor(
                val_dataloader.dataset.tensors[0],
                dtype=torch.float,
                device=self.device,
            )
            treat_val = torch.tensor(
                val_dataloader.dataset.tensors[2],
                dtype=torch.float,
                device=self.device,
            )
            obs_probabilities_val, _, _ = obs_classifier(x_val, treat_val, self.device)

            obs_probabilities_train = obs_probabilities_train[:, :, 0]
            obs_probabilities_val = obs_probabilities_val[:, :, 0]

            # Save model:
            self.obs_classifier = obs_classifier

            # fig, ax = plt.subplots()
            # # sns.kdeplot((val_dataloader.dataset.tensors[3] - pred_intensities_val).flatten().detach().numpy(), fill=0.1, ax=ax)
            # sns.kdeplot((val_dataloader.dataset.tensors[1][:, :, 1] - obs_probabilities_val).flatten().detach().numpy(),
            #             fill=0.1, ax=ax)
            # sns.kdeplot((val_dataloader.dataset.tensors[2] - obs_probabilities_val)[
            #                 val_dataloader.dataset.tensors[2][:, :, 0] == 0].flatten().detach().numpy(), fill=0.1,
            #             ax=ax)
            # sns.kdeplot((val_dataloader.dataset.tensors[3] - pred_intensities_val)[
            #                 val_dataloader.dataset.tensors[2][:, :, 0].isnan()].flatten().detach().numpy(), fill=0.1,
            #             ax=ax)
            # ax.set_title(f"Prediction errors intensities")
            # plt.gcf().set_dpi(300)
            # plt.legend(["No treatments", "Treatments", "Unobserved"])
            # plt.show()

            # Log MSE probability
            true_obs_probabilities_val = torch.tensor(
                val_dataloader.dataset.tensors[5],
                dtype=torch.float,
                device=self.device,
            )
            rmse_obs_prob_val = torch.sqrt(
                torch.mean((true_obs_probabilities_val - obs_probabilities_val).pow(2)))
            print(f"RMSE Validation Intensities: {rmse_obs_prob_val}")

            self.run.log({f"RMSE Validation Intensities": rmse_obs_prob_val})

            # Clip
            obs_probabilities_train[obs_probabilities_train < self.cutoff] = self.cutoff
            obs_probabilities_val[obs_probabilities_val < self.cutoff] = self.cutoff

        ######################
        # Potential outcome prediction

        early_stopping = EarlyStopping(patience=patience, delta=0.0001)

        # create dataloaders
        # observed_only when outcome prediction only (no intensity prediction)
        # Has to be False when Multitask to learn intensities/observation probabilities
        if self.importance_weighting and not self.ground_truth_iw and self.multitask:
            observed_only = False
        else:
            observed_only = True
        if self.importance_weighting and not self.ground_truth_iw and not self.multitask:  # Two-step: do not process weights
            process_weights = False
        else:
            process_weights = True

        train_dataloader, data_shape = self.prepare_dataloader(
            data=train_data,
            batch_size=int(batch_size),
            importance_weights=obs_probabilities_train,
            window=self.window,
            process_weights=process_weights,
            observed_only=observed_only,
        )
        val_dataloader, _ = self.prepare_dataloader(
            data=validation_data,
            batch_size=int(batch_size),
            importance_weights=obs_probabilities_val,
            window=self.window,
            process_weights=process_weights,
            observed_only=observed_only,
        )

        logging.info("Instantiating Neural CDE")

        self.run.log({f"% observed": np.mean(~np.isnan(train_data["cancer_volume"]))})

        model = NeuralCDE(
            input_channels_x=data_shape[2],
            hidden_channels_x=self.hidden_channels_x,
            hidden_channels_enc=self.hidden_channels_enc,
            hidden_layers_enc=self.hidden_layers_enc,
            hidden_channels_dec=self.hidden_channels_dec,
            hidden_layers_dec=self.hidden_layers_dec,
            hidden_channels_map=self.hidden_channels_map,
            hidden_layers_map=self.hidden_layers_map,
            multitask=self.multitask,
            output_channels=1,
            window=self.window,
        )

        print('Number of parameters:', sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))

        model = model.to(self.device)

        learning_rate = 5e-4
        optimizer = torch.optim.Adam([  # All but intensity
            {"params": model.embed_x.parameters()},
            {"params": model.cde_func_encoder.parameters()},
            {"params": model.cde_func_decoder.parameters()},
            {"params": model.outcome.parameters()},
        ], lr=learning_rate, weight_decay=1e-3)
        if self.importance_weighting and not self.ground_truth_iw and self.multitask:  # Special optim for intensity
            optimizer_intensity = torch.optim.Adam(model.intensity.parameters(), lr=learning_rate,
                                                    weight_decay=1e-3)
        else:
            optimizer_intensity = None
        print('Learning rate:', optimizer.param_groups[0]['lr'])

        self.run.watch(model, log="all")

        logging.info("Training CDE")
        epochs = epochs

        for epoch in tqdm(range(epochs)):

            logging.info(f"Training epoch: {epoch}")
            # Train iteration
            model, train_losses_total, train_losses_mse, train_losses_obs = self._train(
                train_dataloader=train_dataloader,
                model=model,
                optimizers=[optimizer, optimizer_intensity],
            )

            # Validation and early stopping check
            logging.info(f"Validation epoch: {epoch}")
            model, val_losses_total = self._test(
                test_dataloader=val_dataloader,
                model=model,
            )

            tqdm.write(
                f"Epoch: {epoch} "
                f"Training loss: {np.average(train_losses_total)}; "
                f"MSE loss: {np.average(train_losses_mse)}; "
                f"Observation loss: {np.average(train_losses_obs)}; "
                f"Val loss: {np.average(val_losses_total)}; "
            )

            self.run.log(
                {
                    "Epoch": epoch,
                    "Training loss": np.average(train_losses_total),
                    "MSE loss": np.average(train_losses_mse),
                    "Observation loss": np.average(train_losses_obs),
                    "Val loss": np.average(val_losses_total),
                },
            )

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            val_loss_epoch = np.average(val_losses_total)
            if np.isnan(val_loss_epoch):
                val_loss_epoch = np.inf
            early_stopping(val_loss_epoch, model)

            if early_stopping.best_score == -np.average(val_losses_total):
                self.model = model

            if early_stopping.early_stop:
                print("Early stopping phase initiated...")
                break

        # Visualize results for validation set:
        val_concat = torch.tensor(val_dataloader.dataset.tensors[0], dtype=torch.float, device=device)
        val_y = torch.tensor(val_dataloader.dataset.tensors[1], dtype=torch.float, device=device)
        val_treat = torch.tensor(val_dataloader.dataset.tensors[2], dtype=torch.float, device=device)
        val_ae = torch.tensor(val_dataloader.dataset.tensors[3], dtype=torch.float, device=device)
        total_obs_prob = torch.tensor(val_dataloader.dataset.tensors[5], dtype=torch.float, device=device)

        pred_y_val, pred_obs_val = self.model(val_concat, val_treat, device)

        outcomes_val = val_y
        ae_val = val_ae

        print('MSE (val):\t\t', torch.mean(torch.square(outcomes_val[ae_val == 1] - pred_y_val[ae_val == 1])).item())
        for i in range(5):
            print('MSE (val)', i, ':\t', torch.mean(torch.square(outcomes_val[:, i][(ae_val == 1)[:, i]] - pred_y_val[:, i][(ae_val == 1)[:, i]])).item())

        # Check intensities:
        if self.importance_weighting and not self.ground_truth_iw and self.multitask:
            obs_prob_val = torch.sigmoid(pred_obs_val)
            rmse_obs_prob_val = torch.sqrt(
                torch.mean((total_obs_prob - obs_prob_val).pow(2)))
            print(f"RMSE Validation Intensities: "f"{rmse_obs_prob_val}")
            self.run.log({f"RMSE Validation Intensities": rmse_obs_prob_val})

        # colors = ['green', 'red', 'black', 'blue', 'orange']
        # max_ids = 5
        # id_counter = 0
        # for i in range(min(4, val_y.shape[0])):
        #     id_counter += 1
        #     # plt.plot(
        #     #     np.arange(val_y.shape[1])[~val_y[i, :].isnan().detach().numpy()],
        #     #     val_y[i, :][~val_y[i, :].isnan()].detach().numpy() * validation_data["inputs_stds"][0] + validation_data["input_means"][0],
        #     #     linestyle='-', color=colors[i % 4], alpha=0.8, marker='o')
        #     plt.plot(
        #         np.arange(outcomes_val.shape[1])[ae_val[i, :] == 1],
        #         outcomes_val[i, :][ae_val[i, :] == 1].detach().numpy() * validation_data["inputs_stds"][0] + validation_data["input_means"][0],
        #         linestyle='-', color=colors[i % 4], alpha=0.8, marker='o')
        #     plt.plot(pred_y_val[i, :].detach().numpy() * validation_data["inputs_stds"][0] + validation_data["input_means"][0],
        #              linestyle='--', color=colors[i % 4], alpha=0.8, marker='.')
        #     if id_counter == max_ids:
        #         break
        #     rmse_instance = torch.mean((outcomes_val[i, :][ae_val[i, :] == 1] - pred_y_val[i, :][ae_val[i, :] == 1]) ** 2)
        #     plt.title(f"True vs Predicted (Validation); RMSE = {rmse_instance}")
        #     plt.gcf().set_dpi(300)
        #     plt.show()

    def predict(self, test_data, label=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Predicting with: {device}")

        # test_processed
        test_concat, test_y, test_treat, total_ae, _, test_obs_prob = self.prepare_dataloader(
            data=test_data,
            return_tensors=True,
            observed_only=False,
        )

        test_concat = torch.tensor(test_concat, dtype=torch.float, device=device)
        test_treat = torch.tensor(test_treat, dtype=torch.float, device=device)
        pred_y_test, pred_obs = self.model(test_concat, test_treat, device)

        test_y = test_y.to(device)
        pred_y_test = pred_y_test.to(device)

        # Evaluate intensities
        if self.importance_weighting and not (self.ground_truth_iw):
            if self.multitask:
                pred_obs = torch.sigmoid(pred_obs)
                mse_intensities = torch.mean((pred_obs - test_obs_prob)**2)
            else:
                pred_obs, _, _ = self.obs_classifier(test_concat, test_treat, device)
                mse_intensities = torch.mean((pred_obs[:, :, 0] - test_obs_prob)**2)
        else:
            mse_intensities = torch.zeros(1)

        # compute norm mse loss - outcomes loss
        mse_test = mse(
            test_y,
            pred_y_test,
            torch.ones_like(test_y, device=device),
        )
        mse_1 = mse(test_y[:, 0], pred_y_test[:, 0], torch.ones_like(test_y[:, 0], device=device))
        mse_2 = mse(test_y[:, 1], pred_y_test[:, 1], torch.ones_like(test_y[:, 1], device=device))
        mse_3 = mse(test_y[:, 2], pred_y_test[:, 2], torch.ones_like(test_y[:, 2], device=device))
        mse_4 = mse(test_y[:, 3], pred_y_test[:, 3], torch.ones_like(test_y[:, 3], device=device))
        mse_5 = mse(test_y[:, 4], pred_y_test[:, 4], torch.ones_like(test_y[:, 4], device=device))

        # Plot some results:
        # linestyles = ['-', '--', '-.', ':', (0, (1, 1))]
        # colors = ['green', 'red', 'black', 'blue', 'orange']
        # max_ids = 5

        # for i in range(min(max_ids, test_y.shape[0])):
        #     plt.plot(test_y[i, :].detach().numpy()
        #              - pred_y_test[i, :].detach().numpy(),     # / 1150,
        #              linestyle=linestyles[i % 4], color=colors[i % 4],
        #              alpha=(1 / test_y.shape[0] + 0.2) / 2, marker='.')
        # plt.hlines(0, 0, test_y.shape[1], colors='gray', alpha=1, linestyles='--', linewidth=2.0)
        # plt.title(f"Prediction errors over time (True - Predicted) ({label})")
        # plt.gcf().set_dpi(300)
        # plt.show()

        # id_counter = 0
        # for i in range(min(max_ids, test_y.shape[0])):
        #     id_counter += 1
        #     plt.plot(test_y[i, :].detach().numpy() * test_data["inputs_stds"][0] + test_data["input_means"][0],
        #              linestyle='-', color=colors[i % 4], alpha=(1 / test_y.shape[0] + 0.2) / 2, marker='None')
        #     plt.plot(pred_y_test[i, :].detach().numpy() * test_data["inputs_stds"][0] + test_data["input_means"][0],
        #              linestyle='None', color=colors[i % 4], alpha=(1 / test_y.shape[0] + 0.2) / 2, marker='.')
        #     if id_counter == max_ids:
        #         break
        # plt.title(f"True vs Predicted ({label})")
        # plt.gcf().set_dpi(300)
        # plt.show()

        # id_counter = 0
        # for i in range(min(4, test_y.shape[0])):
        #     # if active_entries_test[i, :].mean() == 1:
        #     id_counter += 1
        #     plt.plot(test_y[i, :].detach().numpy() * test_data["inputs_stds"][0] + test_data["input_means"][0],
        #              linestyle='-', color=colors[i % 4], alpha=0.8, marker='o')
        #     plt.plot(pred_y_test[i, :].detach().numpy() * test_data["inputs_stds"][0] + test_data["input_means"][0],
        #              linestyle='--', color=colors[i % 4], alpha=0.8, marker='.')
        #     if id_counter == max_ids:
        #         break
        #     rmse_instance = torch.sqrt(torch.mean((test_y[i, :] - pred_y_test[i, :]) ** 2))
        #     plt.title(f"True vs Predicted ({label}); RMSE = {rmse_instance}")
        #     plt.gcf().set_dpi(300)
        #     plt.show()
        #
        # fig, ax = plt.subplots()
        # sns.kdeplot((test_y - pred_y_test).detach().numpy().flatten(), fill=0.1, ax=ax)
        # ax.set_title(f"Prediction errors (True - Predicted) ({label})")
        # plt.gcf().set_dpi(300)
        # plt.show()

        # plt.scatter(test_y.detach().numpy(), pred_y_test.detach().numpy(), marker='.', alpha=0.2)
        # p1 = max(pred_y_test.detach().numpy().max(), test_y.detach().numpy().max())
        # p2 = min(pred_y_test.detach().numpy().min(), test_y.detach().numpy().min())
        # plt.plot([p1, p2], [p1, p2], 'b-')
        # plt.xlabel('True Values', fontsize=15)
        # plt.ylabel('Predictions', fontsize=15)
        # plt.axis('equal')
        # plt.show()

        # Return RMSE (average and per time stamp), MAE, MAPE, Average tumor size per patient (true and pred)
        return np.sqrt(mse_test.item()), np.sqrt(mse_1.item()), np.sqrt(mse_2.item()), np.sqrt(mse_3.item()), np.sqrt(
            mse_4.item()), np.sqrt(mse_5.item()), mse_intensities.item()
