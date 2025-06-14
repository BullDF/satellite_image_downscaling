from ASDM import *


class TransferredModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_layer_norm = nn.BatchNorm1d(num_merra2)

        self.lstm1 = nn.LSTM(num_merra2, 16)
        self.lstm2 = nn.LSTM(16, 32)
        self.lstm3 = nn.LSTM(32, 64)

        self.block1 = BuildingBlock(64, 16)
        self.block2 = BuildingBlock(16, 32)
        self.block3 = BuildingBlock(32, 64)
        self.block4 = BuildingBlock(64, 128)
        self.block5 = BuildingBlock(128, 256, 0.5)
        self.block6 = BuildingBlock(256, 128)
        self.block7 = BuildingBlock(128, 64)
        self.block8 = BuildingBlock(64, 32)
        self.block9 = BuildingBlock(32, 16)
        self.block10 = BuildingBlock(16, num_merra2)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.pre_layer_norm(x)
        x = torch.permute(x, (0, 2, 1))

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)

        return x


class ASDMTE(nn.Module):
    def __init__(self):
        super().__init__()

        self.process_block1 = ProcessBlock1()
        self.temporal_block1 = TemporalBlock()
        self.temporal_block2 = TemporalBlock()
        self.transferred_model = TransferredModel()
        self.transferred_block = BuildingBlock(num_merra2, 16)
        self.process_block2 = ProcessBlock2(16 * 4)

    def forward(self, x):
        general_variables, present, lags = reshape_data(x)

        process_block1_output = self.process_block1(torch.cat([general_variables, present], dim=1))
        temporal_block1_output = self.temporal_block1(lags)
        temporal_block2_output = self.temporal_block2(lags)
        transferred_model_output = self.transferred_model(lags)
        transferred_model_output = self.transferred_block(transferred_model_output)

        combined = torch.cat([process_block1_output, temporal_block1_output, temporal_block2_output, transferred_model_output], dim=1)
        output = self.process_block2(combined)
        return output.squeeze(-1)
