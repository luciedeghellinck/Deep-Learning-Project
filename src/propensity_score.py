from typing import Tuple
import torch

def propensityRegression(dataset: Tuple[torch.Tensor, torch.IntTensor, torch.Tensor]) -> torch.nn.Module:
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

    # logistic regression model
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 1),
        torch.nn.Sigmoid()
    )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train model to predict T given X
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, T)
        loss.backward()
        optimizer.step()

    #  propensity score
    with torch.no_grad():
        prop = model(X)  # .mean().item()

    return prop


