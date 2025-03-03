
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import *
from config import *
import os
from typing import Optional
import matplotlib.pyplot as plt


LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
NUM_EPOCHS = 50


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
    return 1 if num == 1 else num


def train(model: nn.Module,
          train_dataset: TimeSeriesDataset,
          val_dataset: TimeSeriesDataset,
          round: int,
          initial_epoch: int=0,
          checkpoint_round: Optional[int]=None,
          save: bool=True,
          save_every: int=5) -> None:
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

        if save and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), model_dir + f'epoch{epoch + 1}.pth')

@torch.no_grad()
def evaluate(model: nn.Module,
             dataset: TimeSeriesDataset) -> float:
    
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


def plot_losses(train_dataset: TimeSeriesDataset,
                val_dataset: TimeSeriesDataset,
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


if __name__ == '__main__':
    
    train_inputs = np.load(time_series_dir + 'train_inputs.npy')
    train_labels = np.load(time_series_dir + 'train_labels.npy')
    val_inputs = np.load(time_series_dir + 'val_inputs.npy')
    val_labels = np.load(time_series_dir + 'val_labels.npy')

    model = MERRA2Model().to(device)
    train_dataset = TimeSeriesDataset(train_inputs, train_labels)
    val_dataset = TimeSeriesDataset(val_inputs, val_labels)

    train(model, train_dataset, val_dataset, get_training_round(model))

    # load_checkpoint(model, 5, 200)

    # print(evaluate(model, train_dataset))