import torch
from torch import nn


class MLPClassifierBinary(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, 24),
        #     nn.Tanh(),
        #     nn.Linear(24, 16),
        #     nn.Tanh(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )
        self.model = nn.Sequential(
            nn.Linear(input_dim, 24),
            # nn.BatchNorm1d(24),
            # nn.LeakyReLU(0.01),
            nn.Tanh(),
            nn.Dropout(0.3),

            nn.Linear(24, 16),
            # nn.BatchNorm1d(16),
            # nn.LeakyReLU(0.01),
            nn.Tanh(),
            nn.Dropout(0.2),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = self.model(X)
        return out
