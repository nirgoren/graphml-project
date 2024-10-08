import torch
from torch import nn
from torch.utils.data import DataLoader

from data.get_pcpnet_dataset import PU1KTrainDataset


class Trainer:
    def __init__(self, model: nn.Module, batch_size: int):
        self.model = model
        self.batch_size = batch_size

    def get_train_dataloader(self, shuffle=True, normalize=True, **kwargs):
        return DataLoader(
            PU1KTrainDataset(normalize=normalize, **kwargs),
            batch_size=self.batch_size,
            shuffle=shuffle,
            **kwargs,
        )
