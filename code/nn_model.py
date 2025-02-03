
import torch
import torch.nn as nn
from config import checkpoints_dir
from torch.utils.data import Dataset


embed_sizes = {'hour': 16, 'day': 32, 'month': 64, 'season': 128, 'site': 256, 'aridity': 200}
TOTAL_EMBED_SIZE = sum(embed_sizes.values())
NUM_SITES = 11

embed_all_idx = {'month': -1, 'day': -2, 'hour': -3, 'season': -4, 'site': -5}
aridity_idx = {'aridity': -1, 'month': -2, 'day': -3, 'hour': -4, 'season': -5, 'site': -6}


def load_checkpoint(model: nn.Module,
                    round: int,
                    epoch: int) -> None:
    state_dict = torch.load(checkpoints_dir + f'{model.__class__.__name__}{round}/epoch{epoch}.pth', weights_only=True)

    model.load_state_dict(state_dict)


class OpenAQDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class LinearModel(nn.Module):
    def __init__(self, num_vars: int):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(num_vars, 1)
        
    def forward(self, x):
        return self.fc(x)


class SimpleModel(nn.Module):
    def __init__(self, num_vars: int):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(num_vars, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.pre_batch_norm(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    
class ComplexModel(nn.Module):
    def __init__(self, num_vars: int):
        super(ComplexModel, self).__init__()
        self.fc1 = nn.Linear(num_vars, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.pre_batch_norm(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class SimpleLeaky(nn.Module):
    def __init__(self, num_vars: int):
        super(SimpleLeaky, self).__init__()
        self.fc1 = nn.Linear(num_vars, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.LeakyReLU()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.pre_batch_norm(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    

class SimpleTanh(nn.Module):
    def __init__(self, num_vars: int):
        super(SimpleTanh, self).__init__()
        self.fc1 = nn.Linear(num_vars, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars)
        
    def forward(self, x):
        x = self.pre_batch_norm(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.out(x)
        return x
    

class SimpleTanhBN(nn.Module):
    def __init__(self, num_vars: int):
        super(SimpleTanhBN, self).__init__()
        self.fc1 = nn.Linear(num_vars, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.pre_batch_norm(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.tanh(x)
        x = self.out(x)
        return x


class VeryComplexModel(nn.Module):
    def __init__(self, num_vars: int):
        super(VeryComplexModel, self).__init__()
        self.fc1 = nn.Linear(num_vars, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.pre_batch_norm(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.batch_norm5(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    

class EmbeddingModel(nn.Module):
    def __init__(self, num_vars: int):
        super(EmbeddingModel, self).__init__()
        self.site_embedding = nn.Embedding(NUM_SITES, embed_sizes['site'])
        self.fc1 = nn.Linear(num_vars - 1 + embed_sizes['site'], 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars - 1)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.batch_norm3 = nn.BatchNorm1d(512)
        
    def forward(self, x):
        embedded = self.site_embedding(x[:, -1].to(torch.long))
        x = self.pre_batch_norm(x[:, :-1])
        x = torch.concat([x, embedded], dim=1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.out(x)

        return x
    

class SeasonEmbeddingModel(EmbeddingModel):
    def __init__(self, num_vars: int):
        super(SeasonEmbeddingModel, self).__init__(num_vars - 1)
        self.season_embedding = nn.Embedding(4, embed_sizes['season'])
        self.fc1 = nn.Linear(num_vars - 2 + embed_sizes['site'] + embed_sizes['season'], 512)
        
    def forward(self, x):
        site_embedded = self.site_embedding(x[:, -2].to(torch.long))
        season_embedded = self.season_embedding(x[:, -1].to(torch.long))
        x = self.pre_batch_norm(x[:, :-2])
        x = torch.concat([x, site_embedded, season_embedded], dim=1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.out(x)

        return x
    

class EmbedAllModel(nn.Module):
    def __init__(self, num_vars: int):
        super(EmbedAllModel, self).__init__()
        self.site_embedding = nn.Embedding(NUM_SITES, embed_sizes['site'])
        self.season_embedding = nn.Embedding(4, embed_sizes['season'])
        self.month_embedding = nn.Embedding(12, embed_sizes['month'])
        self.day_embedding = nn.Embedding(7, embed_sizes['day'])
        self.hour_embedding = nn.Embedding(24, embed_sizes['hour'])

        self.fc1 = nn.Linear(num_vars - 5 + TOTAL_EMBED_SIZE, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.relu = nn.LeakyReLU()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars - 5)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.batch_norm3 = nn.BatchNorm1d(512)

    def embedding(self, x):
        site_embedded = self.site_embedding(x[:, embed_all_idx['site']].to(torch.long))
        season_embedded = self.season_embedding(x[:, embed_all_idx['season']].to(torch.long))
        month_embedded = self.month_embedding(x[:, embed_all_idx['month']].to(torch.long))
        day_embedded = self.day_embedding(x[:, embed_all_idx['day']].to(torch.long))
        hour_embedded = self.hour_embedding(x[:, embed_all_idx['hour']].to(torch.long))
        return torch.concat([site_embedded, season_embedded, month_embedded, day_embedded, hour_embedded], dim=1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        x = self.pre_batch_norm(x[:, :-5])
        x = torch.concat([x, embedded], dim=1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.out(x)

        return x
    

class AridityModel(nn.Module):
    def __init__(self, num_vars: int):
        super(AridityModel, self).__init__()
        self.site_embedding = nn.Embedding(NUM_SITES, embed_sizes['site'])
        self.season_embedding = nn.Embedding(4, embed_sizes['season'])
        self.month_embedding = nn.Embedding(12, embed_sizes['month'])
        self.day_embedding = nn.Embedding(7, embed_sizes['day'])
        self.hour_embedding = nn.Embedding(24, embed_sizes['hour'])
        self.aridity_embedding = nn.Embedding(2, embed_sizes['aridity'])

        self.fc1 = nn.Linear(num_vars - 6 + TOTAL_EMBED_SIZE, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.pre_batch_norm = nn.BatchNorm1d(num_vars - 6)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        self.batch_norm3 = nn.BatchNorm1d(512)

    def embedding(self, x):
        site_embedded = self.site_embedding(x[:, aridity_idx['site']].to(torch.long))
        season_embedded = self.season_embedding(x[:, aridity_idx['season']].to(torch.long))
        month_embedded = self.month_embedding(x[:, aridity_idx['month']].to(torch.long))
        day_embedded = self.day_embedding(x[:, aridity_idx['day']].to(torch.long))
        hour_embedded = self.hour_embedding(x[:, aridity_idx['hour']].to(torch.long))
        aridity_embedded = self.aridity_embedding(x[:, aridity_idx['aridity']].to(torch.long))
        return torch.concat([site_embedded, season_embedded, month_embedded, day_embedded, hour_embedded, aridity_embedded], dim=1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        x = self.pre_batch_norm(x[:, :-6])
        x = torch.concat([x, embedded], dim=1)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.out(x)

        return x
