# Adapted from: https://github.com/seedatnabeel/TE-CDE/blob/main/src/utils/losses.py

import torch
from torch.nn import functional as F
import math


def mse(ground_truth_outputs, predictions, active_entries, obs_prob=None, norm=1150):
    """
    Computes normed MSE Loss

    Args:
    outputs (torch.tensor): list of true outputs (ground_truth)
    predictions (torch.tensor): list of model predictions
    active_entries (torch.tensor): list of active entries
    norm (int): normalization constant

    Returns:
    mse_loss (float): normed mse loss value
    """
    assert ground_truth_outputs.shape == active_entries.shape
    assert predictions.shape == active_entries.shape
    if obs_prob is None:
        loss = torch.mean(
            (ground_truth_outputs[active_entries == 1.] - predictions[active_entries == 1.]).pow(2)
        )
    else:
        loss = torch.mean(
            (ground_truth_outputs[active_entries == 1.] - predictions[active_entries == 1.]).pow(2)
            / obs_prob[active_entries == 1.]
        )

    return loss


def compute_cross_entropy_loss(outputs, predictions, active_entries):
    """
    Computes cross entropy lossLoss

    Args:
    outputs (torch.tensor): list of true outputs (ground_truth)
    predictions (torch.tensor): list of model predictions
    active_entries (torch.tensor): list of active entries

    Returns:
    ce_loss (float): cross entropy value
    """

    ce_loss = torch.mean(
        -torch.sum(predictions * torch.log(F.softmax(outputs, dim=1)), dim=1),
    )
    return ce_loss
