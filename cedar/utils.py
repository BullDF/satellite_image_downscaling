import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os


data_dir = '../data/'
checkpoints_dir = '../checkpoints/'

indices = {'lat': 0, 'lon': 1, 'season': 2, 'month': 3, 'day_of_week': 4, 'hour': 5}

num_general_variables = len(indices)
num_aerosols = 13
num_meteorology = 12
num_surface_flux = 1
num_merra2 = num_aerosols + num_meteorology + num_surface_flux

idx_aerosols_start = 0
idx_aerosols_end = 325
idx_meteorology_start = 325
idx_meteorology_end = 625
idx_surface_flux_start = 625
idx_surface_flux_end = 650

nums_categories = {'season': 4, 'month': 12, 'day_of_week': 7, 'hour': 24}
total_num_merra2 = 650


class CalibrationDataset(Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = torch.tensor(inputs).to(torch.float32)
        self.labels = torch.tensor(labels).to(torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def load_checkpoint(model: nn.Module,
                    round: int,
                    epoch: int,
                    device: str) -> None:
    state_dict = torch.load(checkpoints_dir + f'{model.__class__.__name__}{round}/epoch{epoch}.pth', weights_only=True, map_location=device)

    model.load_state_dict(state_dict)


def get_training_round(model: nn.Module) -> int:
    files = sorted(os.listdir(checkpoints_dir))
    num = 0
    for file in files:
        if file[:len(model.__class__.__name__)] == model.__class__.__name__ and file[len(model.__class__.__name__):].isnumeric():
            num = max(num, int(file[len(model.__class__.__name__):]) + 1)
        elif num != 0:
            return num
    return 1 if num == 0 else num


def reshape_data(x: torch.Tensor) -> tuple[torch.Tensor]:
    merra2 = x[:, num_general_variables:]
    aerosols = merra2[:, idx_aerosols_start:idx_aerosols_end].view(-1, 25, num_aerosols)
    meteorology = merra2[:, idx_meteorology_start:idx_meteorology_end].view(-1, 25, num_meteorology)
    surface_flux = merra2[:, idx_surface_flux_start:idx_surface_flux_end].view(-1, 25, num_surface_flux)

    general_variables = x[:, :num_general_variables]
    present = torch.cat([aerosols[:, 0, :], meteorology[:, 0, :], surface_flux[:, 0, :]], dim=1)

    aerosols_lags = aerosols[:, 1:, :]
    meteorology_lags = meteorology[:, 1:, :]
    surface_flux_lags = surface_flux[:, 1:, :]

    lags = torch.cat([aerosols_lags, meteorology_lags, surface_flux_lags], dim=2)

    return general_variables, present, lags
