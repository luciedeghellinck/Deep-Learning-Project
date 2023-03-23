import torch
import torch as th
from propensity_score import propensityRegression
from typing import Tuple
from torch.utils.data import DataLoader

def IPM(dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prediction_treated: torch.Tensor,
        prediction_untreated: torch.Tensor):
  """
  Calculates the IPM distance for two probability functions.

  Args:
    dataset: torch tensor where each row represents a person, the first
      column represents the float feature vector, the second is the 0/1
      treatment type and the third is the float outcome.
    prediction_treated:
    prediction_untreated:
  Returns:
    The IPM distance evaluated on all
  """


def weight(dataset: Tuple[th.Tensor, th.Tensor, th.Tensor]) -> th.Tensor:
    """
    Calculates the weight for a given feature vector and a given treatment type.

    Args:
     dataset: torch tensor where each row represents a person, the first
        column represents the float feature vector, the second is the 0/1
        treatment type and the third is the float outcome.
      x: m dimensional float feature vector
      t: 0/1 treatment type.
    Returns:
      The float weight for the feature vector and the treatment type.
    """
    regression = propensityRegression(dataset)
    propensity = regression.forward(dataset[0])
    assert len(propensity.size()) == len(dataset[0].size()) and propensity.size()[-1] == 1

    weight = (dataset[1] * (1 - 2 * propensity) + propensity ** 2) / (propensity * (1 - propensity))

    return weight


def pi(dataset: Tuple[th.Tensor, th.Tensor, th.Tensor], t: int) -> th.Tensor:
    """
    Calculates the percentage of a given treatment type for the dataset.

    Args:
      dataset: torch tensor where each row represents a person, the first
        column represents the float feature vector, the second is the 0/1
        treatment type and the third is the float outcome.
      t: 0/1 treatment type.

    Returns:
      The float percentage of the treatment type.
    """
    pi = th.sum(dataset[1]==t)/dataset[1].size()[0]

    return pi


def adaptedWeight(dataset: Tuple[th.Tensor, th.Tensor, th.Tensor]) -> th.Tensor:
    """
    Calculates the weight for a given feature vector and a given treatment type.

    Args:
     dataset: torch tensor where each row represents a person, the first
        column represents the float feature vector, the second is the 0/1
        treatment type and the third is the float outcome.
      t: 0/1 treatment type.

    Returns:
      The float adapted weight for the feature vector and the treatment type.
    """
    old_weight = weight(dataset)
    pi_0 = pi(dataset, 0)
    pi_1 = pi(dataset, 1)
    adapted_weight = old_weight / 2 * (dataset[1] / pi_1 + (1 - dataset[1]) / pi_0)

    return adapted_weight