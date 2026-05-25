import torch
from torch import nn


class MLPWithIsotonic(nn.Module):
    def __init__(self, mlp_model, iso_layer):
        super().__init__()
        self.mlp = mlp_model
        self.iso = iso_layer

    def forward(self, X):
        probs = self.mlp(X)
        probs_calibrated = self.iso(probs)
        return probs_calibrated
