from functools import lru_cache
from typing import Tuple

import torch

class PropensityRegressorFactory:

    def __init__(self, propensity_dataset):
        self.dataset = propensity_dataset

    def create(self):
        return propensityRegression(
            self.dataset
        )


def propensityRegression(
    dataset: Tuple[torch.Tensor, torch.IntTensor],
    epochs: int = 2000
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

    X, T = dataset

    # u, s, v = torch.pca_lowrank(X, 8)
    # print(u.size(),s.size(),v.size())
    # X = torch.matmul(torch.matmul(u, torch.diag(s)), torch.transpose(v, 0, 1))

    # logistic regression model
    model = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 1), torch.nn.Sigmoid())

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0005)
    lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for _ in range(epochs):
        optimizer.zero_grad()
        t_pred = model(X).squeeze(-1)
        loss = criterion(t_pred, T.float())
        loss.backward()
        optimizer.step()
        lr.step(loss)

    return model
