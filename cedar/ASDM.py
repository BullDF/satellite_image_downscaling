import torch
import torch.nn as nn

from utils import *


class BuildingBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float=0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x


class ProcessBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = BuildingBlock(num_merra2 + num_general_variables, 8)
        self.block2 = BuildingBlock(8, 8)
        self.block3 = BuildingBlock(8, 16)
        self.block4 = BuildingBlock(16, 16)
        self.block5 = BuildingBlock(16, 16)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    

class TemporalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(num_merra2, 8, batch_first=True)
        self.lstm2 = nn.LSTM(8, 16, batch_first=True)
        self.lstm3 = nn.LSTM(16, 32, batch_first=True)
        self.block = BuildingBlock(32, 16)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]
        x = self.block(x)
        return x
    

class ProcessBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = BuildingBlock(16 * 3, 64)
        self.block2 = BuildingBlock(64, 128)
        self.block3 = BuildingBlock(128, 256, 0.5)
        self.block4 = BuildingBlock(256, 128)
        self.block5 = BuildingBlock(128, 64)
        self.block6 = BuildingBlock(64, 32)
        self.block7 = BuildingBlock(32, 16)
        self.block8 = BuildingBlock(16, 8)
        self.block9 = BuildingBlock(8, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        return x
    

class CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.process_block1 = ProcessBlock1()
        self.temporal_block1 = TemporalBlock()
        self.temporal_block2 = TemporalBlock()
        self.process_block2 = ProcessBlock2()

    def forward(self, x):
        general_variables, present, lags = reshape_data(x)

        process_block1_output = self.process_block1(torch.cat([general_variables, present], dim=1))
        temporal_block1_output = self.temporal_block1(lags)
        temporal_block2_output = self.temporal_block2(lags)
        combined = torch.cat([process_block1_output, temporal_block1_output, temporal_block2_output], dim=1)
        output = self.process_block2(combined)
        return output.squeeze(-1)
    
