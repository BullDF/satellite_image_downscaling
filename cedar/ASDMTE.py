from ASDM import *


class TransferredModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_layer_norm = nn.LayerNorm(num_merra2)

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
        x = self.pre_layer_norm(x)

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
