import torch
import torch.nn as nn
from config import checkpoints_dir

from torch.utils.data import Dataset

from model import *


def load_checkpoint(model: nn.Module,
                    round: int,
                    epoch: int,
                    device: str) -> None:
    state_dict = torch.load(checkpoints_dir + f'{model.__class__.__name__}{round}/epoch{epoch}.pth', weights_only=True, map_location=device)

    model.load_state_dict(state_dict)


class CalibrationDataset(Dataset):
    def __init__(self, inputs, labels):
        super().__init__()
        self.inputs = torch.tensor(inputs).to(torch.float32)
        self.labels = torch.tensor(labels).to(torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
    

indices = {'lat': 0, 'lon': 1, 'season': 2, 'month': 3, 'day_of_week': 4, 'hour': 5}
embed_size = 128
nums_categories = {'season': 4, 'month': 12, 'day_of_week': 7, 'hour': 24}


class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleDict({key: nn.Embedding(value, embed_size) for key, value in nums_categories.items()})
    
    def forward(self, x):
        embeddings = [self.embeddings[key](x[:, indices[key]].to(torch.long)) for key in nums_categories.keys()]
        
        x = torch.cat(embeddings, dim=1)
        return x
    

offset = 6
num_aerosols = 13
num_meteorology = 12
num_surface_flux = 1

idx_aerosols_start = 0
idx_aerosols_end = 325
idx_meteorology_start = 325
idx_meteorology_end = 625
idx_surface_flux_start = 625
idx_surface_flux_end = 650

lstm_hidden_size = 128
lstm_num_layers = 3

total_num_merra2 = 650


class LSTMLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_batch_norm_lstm = nn.BatchNorm1d(total_num_merra2)

        self.aerosols_lstm = nn.LSTM(num_aerosols, lstm_hidden_size, batch_first=True, num_layers=lstm_num_layers)
        self.meteorology_lstm = nn.LSTM(num_meteorology, lstm_hidden_size, batch_first=True, num_layers=lstm_num_layers)
        self.surface_flux_lstm = nn.LSTM(num_surface_flux, lstm_hidden_size, batch_first=True, num_layers=lstm_num_layers)
    
    def forward(self, x):
        x = self.pre_batch_norm_lstm(x[:, offset:])

        aerosols = x[:, idx_aerosols_start:idx_aerosols_end].view(-1, 25, num_aerosols)
        aerosols = torch.cat([aerosols[:, 1:, :], aerosols[:, 0, :].unsqueeze(1)], dim=1)

        meteorology = x[:, idx_meteorology_start:idx_meteorology_end].view(-1, 25, num_meteorology)
        meteorology = torch.cat([meteorology[:, 1:, :], meteorology[:, 0, :].unsqueeze(1)], dim=1)

        surface_flux = x[:, idx_surface_flux_start:idx_surface_flux_end].view(-1, 25, num_surface_flux)
        surface_flux = torch.cat([surface_flux[:, 1:, :], surface_flux[:, 0, :].unsqueeze(1)], dim=1)

        aerosols, _ = self.aerosols_lstm(aerosols)
        meteorology, _ = self.meteorology_lstm(meteorology)
        surface_flux, _ = self.surface_flux_lstm(surface_flux)

        x = torch.cat([aerosols[:, -1], meteorology[:, -1], surface_flux[:, -1]], dim=1)

        return x
    

quantiles = [0.5, 0.9, 0.95, 0.99]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.embedding_layer = EmbeddingLayer()
        self.lstm_layer = LSTMLayer()

        self.pre_batch_norm = nn.BatchNorm1d(embed_size * len(nums_categories) + lstm_hidden_size * 3)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(embed_size * len(nums_categories) + lstm_hidden_size * 3 + 2, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)

        self.out = nn.Linear(128, len(quantiles))

    def forward(self, x):
        embedding = self.embedding_layer(x)
        lstm = self.lstm_layer(x)

        pre_layer = torch.cat([embedding, lstm], dim=1)
        pre_layer = self.pre_batch_norm(pre_layer)

        x = torch.cat([x[:, :2], pre_layer], dim=1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        x = self.out(x)
        return x
    