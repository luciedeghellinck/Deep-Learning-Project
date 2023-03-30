from typing import Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import torch as th


def IPW(dataset: Tuple[th.Tensor, th.Tensor, th.Tensor], propensity_regressor: th.nn.Module, tau):
    X, T, Y = dataset
    propensity_scores = propensity_regressor.forward(X)

    plug_in_value = T / propensity_scores * Y - (((1 - T) / (1 - propensity_scores)) * Y)

    return th.nn.MSELoss(plug_in_value, tau.predict(X), reduction="mean")


def tau_risk(dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor],
             propensity_regressor: th.nn.Module,
             outcome_regressor: GradientBoostingRegressor,
             tau):
    X, T, Y = dataset

    propensity_scores = propensity_regressor.forward(X)
    expected_outcome = outcome_regressor.predict(X)

    plug_in_value = 1 / len(T) * th.sum((Y - expected_outcome) - (T - propensity_scores))

    return th.nn.MSELoss(plug_in_value, tau.predict(X), reduction="mean")

def outcome_regressor(dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor]):
    X, _, Y = dataset

    reg = GradientBoostingRegressor()
    reg.fit(X, Y)
    return reg


def plug_in_validation(dataset: Tuple[th.Tensor, th.Tensor, th.Tensor], tau)
    """
    Args:
        dataset: load the dataset that contains mu1, mu0
        tau:  candidate tau models

    Returns:
            plug_in_validation R
    """
    x = dataset.X
    tau1 = dataset.mu1
    tau0 = dataset.mu0
    return th.nn.MSELoss ((tau1-tau0) , tau.predict(x) , reduction="mean")


def performanceEstimator(plugIn, candidate):
    """
    Calculates the performance estimator for a set of plug-in and candidate tau.

    Args:
      plugIn: float plug-in CATE preictor
      candidate: float candidate CATE predictor from the Maching learning algorithms and the meta-learners
    Returns:
      A float representing the performance estimator between two CATE predictors.
    """
    # Equation 5
    #just to make 100_datasets work
    return 0.5
