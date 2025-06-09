import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from quantile_model import *

import os

import argparse

from config import *


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def get_training_round(model: nn.Module) -> int:
    files = sorted(os.listdir(checkpoints_dir))
    num = 0
    for file in files:
        if file[:len(model.__class__.__name__)] == model.__class__.__name__ and file[len(model.__class__.__name__):].isnumeric():
            num = max(num, int(file[len(model.__class__.__name__):]) + 1)
        elif num != 0:
            return num
    return 1 if num == 0 else num


def quantile_loss(preds, target, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(torch.mean(loss))
    return sum(losses)


def train(model: nn.Module,
          train_dataset: CalibrationDataset,
          val_dataset: CalibrationDataset,
          round: int,
          num_epochs: int,
          lr: float,
          train_bs: int,
          eval_bs: int,
          initial_epoch: int,
          checkpoint_round: int,
          save: bool,
          save_every: int) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = quantile_loss
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4)

    if initial_epoch != 0:
        load_checkpoint(model, checkpoint_round, initial_epoch, device)

    if save:
        model_dir = checkpoints_dir + f'{model.__class__.__name__}{round}/'
        os.makedirs(model_dir)

    print()
    print(f'Epoch {initial_epoch}/{num_epochs + initial_epoch}, Training Loss: {evaluate(model, train_dataset, eval_bs)}')
    print(f'Validation Loss: {evaluate(model, val_dataset, eval_bs)}')

    if save:
        torch.save(model.state_dict(), model_dir + f'epoch{initial_epoch}.pth')

    for epoch in range(initial_epoch, num_epochs + initial_epoch):
        print()
        print(f'Training epoch {epoch + 1}')
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels, quantiles)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f'Epoch {epoch + 1}/{num_epochs + initial_epoch}, Training Loss: {evaluate(model, train_dataset, eval_bs)}')
        print(f'Validation Loss: {evaluate(model, val_dataset, eval_bs)}')

        if save and (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), model_dir + f'epoch{epoch + 1}.pth')

@torch.no_grad()
def evaluate(model: nn.Module,
             dataset: CalibrationDataset,
             bs: int) -> float:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=bs, num_workers=4)
    criterion = nn.MSELoss().to(device)
    running_loss = 0.0
    running_quantile_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        preds = outputs[:, -1]
        loss = criterion(preds, labels)
        running_loss += loss.item()
        running_quantile_loss += quantile_loss(outputs, labels, quantiles).item()
    return running_loss / len(dataloader), running_quantile_loss / len(dataloader)


def load_data() -> tuple[CalibrationDataset]:
    train_inputs = np.load(data_dir + 'train_inputs.npy')
    train_labels = np.load(data_dir + 'train_labels.npy')
    val_inputs = np.load(data_dir + 'val_inputs.npy')
    val_labels = np.load(data_dir + 'val_labels.npy')

    train_dataset = CalibrationDataset(train_inputs, train_labels)
    val_dataset = CalibrationDataset(val_inputs, val_labels)

    return train_dataset, val_dataset


if __name__ == '__main__':
    print(f'Using {device}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_bs', type=int, default=512, help='Training batch size')
    parser.add_argument('--eval_bs', type=int, default=2048, help='Evaluation batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--initial_epoch', type=int, default=0, help='Initial training epoch')
    parser.add_argument('--checkpoint_round', type=int, default=0, help='Checkpoint round to load from')
    parser.add_argument('--no_save', dest='save', action='store_false', help='Do not save model checkpoints')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    parser.set_defaults(save=True)
    args = parser.parse_args()

    model = CalibrationQuantileModel().to(device)
    train_dataset, val_dataset = load_data()

    lr = args.lr
    train_bs = args.train_bs
    eval_bs = args.eval_bs
    num_epochs = args.epochs
    initial_epoch = args.initial_epoch
    checkpoint_round = args.checkpoint_round
    save = args.save
    save_every = args.save_every

    print('Training with parameters:')
    print(f'Learning Rate: {lr}')
    print(f'Training Batch Size: {train_bs}')
    print(f'Evaluation Batch Size: {eval_bs}')
    print(f'Number of Epochs: {num_epochs}')
    print(f'Initial Epoch: {initial_epoch}')
    if initial_epoch:
        print(f'Checkpoint Round: {checkpoint_round}')
    print(f'Save Checkpoints: {save}')
    if save:
        print(f'Save Every: {save_every}')

    train(model, train_dataset, val_dataset, get_training_round(model), num_epochs, lr, train_bs, eval_bs, initial_epoch, checkpoint_round, save, save_every)

    # load_checkpoint(model, 5, 200)

    # print(evaluate(model, train_dataset))
