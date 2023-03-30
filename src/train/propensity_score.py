from functools import lru_cache
from typing import Tuple
import torch


def propensityRegression(
    dataset: Tuple[torch.Tensor, torch.IntTensor, torch.Tensor],
    validation: Tuple[torch.Tensor, torch.IntTensor, torch.Tensor],
) -> torch.nn.Module:
    """
    Evaluates a regression function for the propensity given the known propensity scores

    Args:
      dataset: torch tensor where the first Tensor represents a feature vector, the second is the 0/1
        treatment type and the third is the tensor outcome.

    Returns:
      A function that takes an m dimensional feature vector as input and
      estimates its propensity
    """
    #
    # prop_model = LogisticRegression().fit(dataset.X, dataset.T)
    # prop_score = prop_model.predict proba(dataset.X)[:，1]
    # return prop_score
    # Call propensityScore to use the propensity for each input feature vector

    X, T, _ = dataset
    X_val, T_val, _ = validation

    # logistic regression model
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1), torch.nn.Sigmoid())

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model to predict T given X
    current_patience = 0
    patience = 5
    lowest_loss = float("inf")
    while current_patience < patience:
        optimizer.zero_grad()
        t_pred = model(X).squeeze(-1)
        loss = criterion(t_pred, T.float())
        loss.backward()
        optimizer.step()

        validation_loss = criterion(model(X_val).squeeze(-1), T_val.float())

        if validation_loss >= lowest_loss:
            current_patience += 1
        else:
            current_patience = 0
            lowest_loss = validation_loss

    return model
