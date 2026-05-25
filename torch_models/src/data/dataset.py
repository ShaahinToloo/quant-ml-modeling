import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]
