import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import *


embed_size = 128
nums_categories = {'season': 4, 'month': 12, 'day_of_week': 7, 'hour': 24}
indices = {'lat': 0, 'lon': 1, 'season': 2, 'month': 3, 'day_of_week': 4, 'hour': 5}


class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleDict({key: nn.Embedding(value, embed_size) for key, value in nums_categories.items()})
    
    def forward(self, x):
        embeddings = [self.embeddings[key](x[:, indices[key]].to(torch.long)) for key in nums_categories.keys()]
        
        x = torch.cat(embeddings, dim=1)
        return x
    

class Transformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)

    def forward(self, x):
        x = self.encoder(x)
        return x
    

num_general_variables = 6
num_aerosols = 13
num_meteorology = 12
num_surface_flux = 1

idx_aerosols_start = 0
idx_aerosols_end = 325
idx_meteorology_start = 325
idx_meteorology_end = 625
idx_surface_flux_start = 625
idx_surface_flux_end = 650


class TransformerCalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.aerosols_transformer = Transformer(num_aerosols)
        self.meteorology_transformer = Transformer(num_meteorology)
        self.surface_flux_transformer = Transformer(num_surface_flux)

        self.linear1 = nn.Linear(2 + embed_size * len(nums_categories) + num_aerosols + num_meteorology + num_surface_flux, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()

    def forward(self, x):
        embedding = self.embedding(x)

        merra2 = x[:, num_general_variables:]

        aerosols = merra2[:, idx_aerosols_start:idx_aerosols_end].view(-1, 25, num_aerosols)
        aerosols = torch.cat([aerosols[:, 1:, :], aerosols[:, 0, :].unsqueeze(1)], dim=1)
        aerosols_output = self.aerosols_transformer(aerosols)

        meteorology = x[:, idx_meteorology_start:idx_meteorology_end].view(-1, 25, num_meteorology)
        meteorology = torch.cat([meteorology[:, 1:, :], meteorology[:, 0, :].unsqueeze(1)], dim=1)
        meteorology_output = self.meteorology_transformer(meteorology)

        surface_flux = x[:, idx_surface_flux_start:idx_surface_flux_end].view(-1, 25, num_surface_flux)
        surface_flux = torch.cat([surface_flux[:, 1:, :], surface_flux[:, 0, :].unsqueeze(1)], dim=1)
        surface_flux_output = self.surface_flux_transformer(surface_flux)

        x = torch.cat([x[:, indices['lat']].unsqueeze(1), x[:, indices['lon']].unsqueeze(1), embedding, aerosols_output[:, -1, :], meteorology_output[:, -1, :], surface_flux_output[:, -1, :]], dim=1)
        
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    