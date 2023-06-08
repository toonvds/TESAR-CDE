# Adapted from: https://github.com/patrick-kidger/torchcde
# Adapted from: https://github.com/seedatnabeel/TE-CDE/blob/main/src/models/CDE_model.py

import logging

import torch
import torch.nn as nn
import torchcde


######################
# A CDE model is defined as
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_channels_func, hidden_layers_func):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        layers = [torch.nn.Linear(hidden_channels, hidden_channels_func)]
        layers.append(torch.nn.ReLU())
        for i in range(hidden_layers_func):
            layers.append(torch.nn.Linear(hidden_channels_func, hidden_channels_func))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_channels_func, input_channels * hidden_channels))
        self.func = nn.Sequential(*layers)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)

        z = self.func(z)

        # z = self.single_layer(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()

        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)

        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(
            self,
            input_channels_x,
            hidden_channels_x,
            hidden_channels_enc,
            hidden_layers_enc,
            hidden_channels_dec,
            hidden_layers_dec,
            hidden_channels_map,
            hidden_layers_map,
            output_channels,    # Per timestep
            multitask,
            window=7,
            interpolation="cubic",
            # interpolation="linear",
            prediction="regression",
            invariance=True
    ):
        super(NeuralCDE, self).__init__()

        # Linear embedding layer
        self.embed_x = torch.nn.Sequential(
            torch.nn.Linear(input_channels_x, hidden_channels_x),
        )

        # Function
        self.cde_func_encoder = CDEFunc(input_channels_x, hidden_channels_x, hidden_channels_enc, hidden_layers_enc)
        self.cde_func_decoder = CDEFunc(4, hidden_channels_x, hidden_channels_dec, hidden_layers_dec)

        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels_x, hidden_channels_map),
                torch.nn.ReLU(),
            )

        # Linear output layer
        if prediction == "regression":
            if hidden_layers_map == 1:
                self.outcome = nn.Sequential(torch.nn.Linear(hidden_channels_x, output_channels))
            else:
                layers = []
                layers.append(torch.nn.Linear(hidden_channels_x, hidden_channels_map))
                layers.append(torch.nn.ReLU())
                for i in range(hidden_layers_map - 2):
                    layers.append(torch.nn.Linear(hidden_channels_map, hidden_channels_map))
                    layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(hidden_channels_map, output_channels))
                self.outcome = nn.Sequential(*layers)

            if multitask:
                if hidden_layers_map == 1:
                    self.intensity = nn.Sequential(torch.nn.Linear(hidden_channels_x, output_channels))
                else:
                    layers = []
                    layers.append(torch.nn.Linear(hidden_channels_x, hidden_channels_map))
                    layers.append(torch.nn.ReLU())
                    for i in range(hidden_layers_map - 2):
                        layers.append(torch.nn.Linear(hidden_channels_map, hidden_channels_map))
                        layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Linear(hidden_channels_map, output_channels))
                    self.intensity = nn.Sequential(*layers)
        elif prediction == "classification":
            if hidden_layers_map == 1:
                self.outcome = nn.Sequential(torch.nn.Linear(hidden_channels_x, output_channels))
            else:
                layers = []
                layers.append(torch.nn.Linear(hidden_channels_x, hidden_channels_map))
                layers.append(torch.nn.ReLU())
                for i in range(hidden_layers_map - 2):
                    layers.append(torch.nn.Linear(hidden_channels_map, hidden_channels_map))
                    layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(hidden_channels_map, output_channels))
                self.outcome = nn.Sequential(*layers)

        self.hidden_channels_x = hidden_channels_x

        self.interpolation = interpolation
        self.window = window
        self.prediction = prediction
        self.multitask = multitask

        logging.info(f"Interpolation type: {self.interpolation}")

    def forward(self, x, treat, device):
        # Interpolation:
        if self.interpolation == "cubic":
            x = torchcde.NaturalCubicSpline(x)
            treat = torchcde.NaturalCubicSpline(treat)
        elif self.interpolation == "linear":
            # print('Adjust in prepare_dataloader! Cubic used now')
            x = torchcde.LinearInterpolation(x)
            treat = torchcde.LinearInterpolation(treat)
        else:
            raise ValueError(
                "Only 'linear' and 'cubic' interpolation methods are implemented.",
            )

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        z_0 = torch.tensor(
            self.embed_x(x.evaluate(x.interval[0])),
            dtype=torch.float,
            device=device,
        )

        ######################
        # Solve the CDE.
        ######################
        # t =x.interval adds the time tracking component to the CDE
        z_t = torch.tensor(
            torchcde.cdeint(
                X=x,
                z0=z_0,
                func=self.cde_func_encoder,
                t=x.interval,
                adjoint=True,
                method="euler",
                options=dict(step_size=1.)      # Default -- This works good given discrete data
            ), dtype=torch.float, device=device)

        ######################
        # Both the initial value and the terminal value are returned from cdeint;
        # Extract just the terminal value and then apply a linear map.
        ######################
        z_t = z_t[:, -1, :]

        z_s = torch.tensor(
            torchcde.cdeint(
                X=treat,
                z0=z_t,
                func=self.cde_func_decoder,
                t=treat.grid_points,
                adjoint=True,
                method="euler",
                options=dict(step_size=1.)
            ), dtype=torch.float, device=device)

        # Get predictions based on z_hat:
        if self.prediction == "regression":
            pred_y = self.outcome(z_s)

            # Adjust shapes because of rectilinear
            if pred_y.shape[2] == 1:
                pred_y = pred_y.flatten(1)

            if self.multitask:
                pred_obs = self.intensity(z_s)
                if pred_obs.shape[2] == 1:
                    pred_obs = pred_obs.flatten(1)
            else:
                pred_obs = None

            return pred_y, pred_obs

        elif self.prediction == "classification":
            pred_obs = self.outcome(z_s)

            pred_obs_sigmoid = torch.sigmoid(pred_obs)

            return pred_obs_sigmoid, pred_obs, pred_obs_sigmoid  # int_obs[:, -1]

        else:
            return NotImplementedError("Only regression and classification are implemented")
