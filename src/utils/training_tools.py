# Adapted from: https://github.com/seedatnabeel/TE-CDE/blob/main/src/utils/training_tools.py

import numpy as np
import torch


class EarlyStopping:
    """Early stopping: stop training if validation loss doesn't improve after n patience steps"""

    def __init__(self, patience=5, delta=0.001, path="checkpoint.pt"):
        """
        Args:
            patience (int): Patience in epochs
            delta (float): minimum delta change in validation loss
            path (str): path for model checkpoints
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        # first epoch
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        # update counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # save best model
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""

        print(
            f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...",
        )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def enable_dropout(model):
    """Enables dropout at test time - for Monte Carlo Dropout"""
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()
    return model
