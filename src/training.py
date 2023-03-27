import torch as th
from torch.utils.data import DataLoader


def fit(train_loader: DataLoader,
        test_loader: DataLoader,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        criterion: th.nn.Module,
        early_stopping_patience: int):
    min_val_loss = float('inf')
    patience_count = 0
    while patience_count < early_stopping_patience:
        train_loss, accuracy = train(train_loader,
                                     model,
                                     optimizer,
                                     criterion)
        print(f"train accuracy: {accuracy}")
        print(f"train loss: {train_loss}")
        val_loss, accuracy = test(test_loader,
                                  model,
                                  criterion)
        print(f"val accuracy: {accuracy}")
        print(f"val loss: {val_loss}")
        if val_loss > min_val_loss:
            early_stopping_patience += 1

        min_val_loss = min(min_val_loss, val_loss)


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
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        _, predicted = th.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total


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
    correct = 0
    total = 0

    # Use torch.no_grad to skip gradient calculation, not needed for evaluation
    with th.no_grad():
        # iterate through batches
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = th.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return avg_loss / len(test_loader), 100 * correct / total