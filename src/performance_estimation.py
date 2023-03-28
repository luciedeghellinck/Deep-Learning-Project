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





def candidatePredictorTau(x, MLAlgorithm, metalearner):
    """
    Calculates the tau for a given a Machine Learning algorithm, a meta-learner and a feature vector.

    Args:
      x: m dimensional float feature vector.
      MLAlgorithm: Machine-learning algorithm.
      metaLearner: metal-learner.
    Returns:
      A float representing the CATE predictor for the algorithm and the meta-learner for the feature.
    """
    # If-else function that checks the machine learning algorithm and the metalearner type
    pass


def performanceEstimator(plugIn, candidate, features: th.Tensor):
    """
    Calculates the performance estimator for a set of plug-in and candidate tau.

    Args:
      plugIn: float plug-in CATE preictor
      candidate: float candidate CATE predictor from the Maching learning algorithms and the meta-learners
    Returns:
      A float representing the performance estimator between two CATE predictors.
    """
    # Equation 5
    pass