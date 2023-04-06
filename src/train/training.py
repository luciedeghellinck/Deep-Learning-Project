from collections import Callable

import torch as th
from torch.utils.data import DataLoader


def fit(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model: th.nn.Module,
    optimizer: th.optim.Optimizer,
    criterion: th.nn.Module,
    patience: int = 5,
):
    curr_patience = 0
    min_val_loss = float('inf')
    while curr_patience <= patience:
        train_loss = train(train_loader, model, optimizer, criterion)
        print(f"train loss: {train_loss}")
        val_loss = test(test_loader, model, criterion)
        print(f"val loss: {val_loss}")
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            curr_patience = 0
        else:
            curr_patience += 1


# Copy and paste from the assignements --> check this
# Is this where we have to input equation 5?
def train(train_loader, model, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        model: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """
    model.train()
    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        optimizer.zero_grad()
        loss = criterion(data, model)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        total += data[0].size(0)
    else:
        i = 1

    return avg_loss / i


def test(test_loader, model, criterion):
    """
    Evaluates network in batches.

    Args:
        test_loader: Data loader for test set.
        model: Neural network model.
        criterion: Loss function (e.g. cross-entropy loss).
    """
    model.eval()
    avg_loss = 0
    total = 0
    # iterate through batches
    for i, data in enumerate(test_loader):
        # zero the parameter gradients
        loss = criterion(data, model)

        # keep track of loss and accuracy
        avg_loss += loss
        total += data[0].size(0)
    else:
        i = 1

    return avg_loss / i
