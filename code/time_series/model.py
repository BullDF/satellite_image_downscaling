import torch
import torch.nn as nn
from config import checkpoints_dir

from torch.utils.data import Dataset

nums_variables = {'aerosols': 13, 'meteorology': 12, 'surface_flux': 1}
indices = {'site': 0, 'lat': 1, 'lon': 2, 'year': 3, 'month': 4, 'day': 5, 'hour': 6, 'aerosols_start': 7, 'aerosols_end': 7 + 13 * 25, 'meteorology_start': 7 + 13 * 25, 'meteorology_end': 7 + 13 * 25 + 12 * 25, 'surface_flux_start': 7 + 13 * 25 + 12 * 25, 'surface_flux_end': 7 + 13 * 25 + 12 * 25 + 1 * 25}
num_sites = 11


def load_checkpoint(model: nn.Module,
                    round: int,
                    epoch: int) -> None:
    state_dict = torch.load(checkpoints_dir + f'{model.__class__.__name__}{round}/epoch{epoch}.pth', weights_only=True)

    model.load_state_dict(state_dict)


class TimeSeriesDataset(Dataset):
    
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = torch.tensor(inputs).to(torch.float32)
        self.labels = torch.tensor(labels).to(torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    

class SimpleLSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.aerosols_lstm = nn.LSTM(nums_variables['aerosols'], 64, batch_first=True)
        self.meteorology_lstm = nn.LSTM(nums_variables['meteorology'], 64, batch_first=True)
        self.surface_flux_lstm = nn.LSTM(nums_variables['surface_flux'], 64, batch_first=True)

        self.pre_batch_norm = nn.BatchNorm1d(650)
        self.after_lstm_batch_norm = nn.BatchNorm1d(64 * 3)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc1_batch_norm = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pre_batch_norm(x)

        aerosols = x[:, indices['aerosols_start']:indices['aerosols_end']].view(-1, 25, nums_variables['aerosols'])
        meteorology = x[:, indices['meteorology_start']:indices['meteorology_end']].view(-1, 25, nums_variables['meteorology'])
        surface_flux = x[:, indices['surface_flux_start']:indices['surface_flux_end']].view(-1, 25, nums_variables['surface_flux'])

        aerosols, _ = self.aerosols_lstm(aerosols)
        meteorology, _ = self.meteorology_lstm(meteorology)
        surface_flux, _ = self.surface_flux_lstm(surface_flux)

        x = torch.cat([aerosols[:, -1], meteorology[:, -1], surface_flux[:, -1]], dim=1)
        x = self.after_lstm_batch_norm(x)

        x = self.fc1(x)
        x = self.fc1_batch_norm(x)
        x = self.relu(x)
        x = self.out(x)

        return x