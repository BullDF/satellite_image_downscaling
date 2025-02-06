
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from nn_model import *
from config import *
import os
from typing import Optional, List
import matplotlib.pyplot as plt


LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 4096
NUM_EPOCHS = 100


if torch.cuda.is_available():
    device = 'cuda'
    print('Using cuda')
elif torch.backends.mps.is_available():
    device = 'mps'
    print('Using mps')
else:
    device = 'cpu'
    print('Using cpu')


def get_training_round(model: nn.Module) -> int:
    files = sorted(os.listdir(checkpoints_dir))
    num = 0
    for file in files:
        if file[:len(model.__class__.__name__)] == model.__class__.__name__ and file[len(model.__class__.__name__):].isnumeric():
            num = max(num, int(file[len(model.__class__.__name__):]) + 1)
        elif num != 0:
            return num
    return 1


def train(model: nn.Module,
          train_dataset: OpenAQDataset,
          val_dataset: OpenAQDataset,
          round: int,
          initial_epoch: int=0,
          checkpoint_round: Optional[int]=None,
          save: bool=True) -> None:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss().to(device)
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    if initial_epoch != 0:
        load_checkpoint(model, checkpoint_round, initial_epoch)

    if save:
        model_dir = checkpoints_dir + f'{model.__class__.__name__}{round}/'
        os.makedirs(model_dir)

    print(f'Epoch {initial_epoch}/{NUM_EPOCHS + initial_epoch}, Training Loss: {evaluate(model, train_dataset)}')
    print(f'Validation Loss: {evaluate(model, val_dataset)}')

    if save:
        torch.save(model.state_dict(), model_dir + f'epoch{initial_epoch}.pth')

    for epoch in range(initial_epoch, NUM_EPOCHS + initial_epoch):
        print(f'Training epoch {epoch + 1}')
        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS + initial_epoch}, Training Loss: {evaluate(model, train_dataset)}')
        print(f'Validation Loss: {evaluate(model, val_dataset)}')

        if (epoch + 1) % 5 == 0 and save:
            torch.save(model.state_dict(), model_dir + f'epoch{epoch + 1}.pth')

@torch.no_grad()
def evaluate(model: nn.Module,
             dataset: OpenAQDataset) -> float:
    
    model.eval()
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    criterion = nn.MSELoss().to(device)
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    return running_loss / len(dataloader)


def plot_losses(train_dataset: OpenAQDataset,
                val_dataset: OpenAQDataset,
                model: nn.Module,
                round: int,
                initial_epoch: int,
                final_epoch: int) -> None:
    
    model = model.to(device)
    train_losses = []
    val_losses = []
    epochs = list(range(initial_epoch, final_epoch + 5, 5))

    for epoch in tqdm(epochs):
        load_checkpoint(model, round, epoch)

        train_losses.append(evaluate(model, train_dataset))
        val_losses.append(evaluate(model, val_dataset))

    
    plt.plot(epochs, train_losses, label='Training')
    plt.plot(epochs, val_losses, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model.__class__.__name__} Training and Validation Losses')
    plt.show()


def leave_one_out_cv(X_train: np.ndarray,
                     y_train: np.ndarray,
                     num_folds: int) -> List[nn.Module]:
    models = []
    for k in range(num_folds):
        print(f'Leaving out site {k}')
        leftout_idx = X_train[:, aridity_idx['site']] == k
        X_train_keep = X_train[~leftout_idx]
        X_train_leftout = X_train[leftout_idx]
        y_train_keep = y_train[~leftout_idx]
        y_train_leftout = y_train[leftout_idx]

        model = AridityModel(len(X_train_keep[0])).to(device)
        train_dataset = OpenAQDataset(X_train_keep, y_train_keep)
        val_dataset = OpenAQDataset(X_train_leftout, y_train_leftout)
        train(model, train_dataset, val_dataset, 0, save=False)

        models.append(model)

    return models


if __name__ == '__main__':
    # X_train = np.load(nn_data_dir + 'merra2_only/X_train.npy')
    # y_train = np.load(nn_data_dir + 'merra2_only/y_train.npy')
    # X_val = np.load(nn_data_dir + 'merra2_only/X_val.npy')
    # y_val = np.load(nn_data_dir + 'merra2_only/y_val.npy')

    # X_train = np.load(nn_data_dir + 'cyclical_encoding/X_train_time.npy')
    # y_train = np.load(nn_data_dir + 'cyclical_encoding/y_train_time.npy')
    # X_val = np.load(nn_data_dir + 'cyclical_encoding/X_val_time.npy')
    # y_val = np.load(nn_data_dir + 'cyclical_encoding/y_val_time.npy')

    # X_train = np.load(nn_data_dir + 'time_loc/X_train_time_loc.npy')
    # y_train = np.load(nn_data_dir + 'time_loc/y_train_time_loc.npy')
    # X_val = np.load(nn_data_dir + 'time_loc/X_val_timghe_loc.npy')
    # y_val = np.load(nn_data_dir + 'time_loc/y_val_time_loc.npy')

    # X_train = np.load(nn_data_dir + 'loc_embed/X_train_loc_embed.npy')
    # y_train = np.load(nn_data_dir + 'loc_embed/y_train_loc_embed.npy')
    # X_val = np.load(nn_data_dir + 'loc_embed/X_val_loc_embed.npy')
    # y_val = np.load(nn_data_dir + 'loc_embed/y_val_loc_embed.npy')

    # X_train = np.load(nn_data_dir + 'season/X_train_season.npy')
    # y_train = np.load(nn_data_dir + 'season/y_train_season.npy')
    # X_val = np.load(nn_data_dir + 'season/X_val_season.npy')
    # y_val = np.load(nn_data_dir + 'season/y_val_season.npy')

    # X_train = np.load(nn_data_dir + 'embed_all/X_train_embed_all.npy')
    # y_train = np.load(nn_data_dir + 'embed_all/y_train_embed_all.npy')
    # X_val = np.load(nn_data_dir + 'embed_all/X_val_embed_all.npy')
    # y_val = np.load(nn_data_dir + 'embed_all/y_val_embed_all.npy')

    # ------------------------------------------------------------------

    # X_train = np.load(nn_data_dir + 'aridity/X_train_aridity.npy')
    # y_train = np.load(nn_data_dir + 'aridity/y_train_aridity.npy')
    # X_val = np.load(nn_data_dir + 'aridity/X_val_aridity.npy')
    # y_val = np.load(nn_data_dir + 'aridity/y_val_aridity.npy')

    # ------------------------------------------------------------------

    X_train = np.load(nn_data_dir + 'removed_multicollinearity/X_train_removed_multicollinearity.npy')
    y_train = np.load(nn_data_dir + 'removed_multicollinearity/y_train_removed_multicollinearity.npy')
    X_val = np.load(nn_data_dir + 'removed_multicollinearity/X_val_removed_multicollinearity.npy')
    y_val = np.load(nn_data_dir + 'removed_multicollinearity/y_val_removed_multicollinearity.npy')

    # model = AridityModel(len(X_train[0])).to(device)
    model = AridityModelBatchNormAfterActivation(len(X_train[0])).to(device)
    train_dataset = OpenAQDataset(X_train, y_train)
    val_dataset = OpenAQDataset(X_val, y_val)

    train(model, train_dataset, val_dataset, get_training_round(model), save=False)

    # load_checkpoint(model, 5, 200)

    # print(evaluate(model, train_dataset))

    # leave_one_out_cv(X_train, y_train, 11)