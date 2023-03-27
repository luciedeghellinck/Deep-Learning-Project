import torch as th
import torch.nn as nn
from scipy.stats import spearmanr


def Rtrue(tauHat: th.Tensor, tauTilde: th.Tensor) -> th.Tensor:
    """
    Calculates the performance metric R_true as defined in Equation 1

    Args:
        tauHat: 1D tensor of size n containing the CATE prediction using a specific ML algorithm and meta-learner
        tauTilde: 1D tensor of size n containing the prediction using the plug in tau
    Returns:
        The performance metric as defined in Equation 1
    """
    squared_error = (tauHat - tauTilde) ** 2
    expected = th.mean(squared_error)

    return expected


def regret(tauHatSelected: th.Tensor, tauHatBest: th.Tensor, tauTilde: th.Tensor):
    """
    Calculates the regret: the difference between the true performance of the selected model and that of the best
    possible candidate in M

    Args:
        tauHatSelected: 1D tensor of size n containing the best CATE prediction calculated using the validation set
        tauHatBest: 1D tensor of size n containing the best CATE prediction calculated using the test set
    Returns:
        The regret between the two CATE predictors
    """

    return (Rtrue(tauHatSelected, tauTilde) - Rtrue(tauHatBest, tauTilde)) / Rtrue(tauHatBest, tauTilde)


def rankCorrelation(tauHatSelected: th.Tensor, tauHatBest: th.Tensor):
    """
    Calculates the Spearman Rank Correlation

    Args:
        tauHatSelected: 1D tensor of size n containing the best CATE prediction calculated using the validation set
        tauHatBest: 1D tensor of size n containing the best CATE prediction calculated using the test set
    Returns:
        The Spearman Rank Correlation between the two CATE predictors
    """
    correlation = spearmanr(tauHatSelected, tauHatBest)
    return correlation.pvalue()


def NRMSE(tauHatSelected: th.Tensor, tauHatBest: th.Tensor):


    # """
    # Calculates the regret: the difference between the true performance of the selected model and that of the best
    # possible candidate in M
    #
    # Args:
    #     tauHat: 5 x 5 x n tensor where each element is the CATE predictor for the candidate model (the line represents an
    #     algorithm, and the column the meta learner), the depth contains the tau for the individual input feature sets
    #     (n input X)
    #     tauTilde: tensor of size n representing the CATE predictor predicted by the network.
    # Returns:
    #     The regret for the dataset.
    # """
