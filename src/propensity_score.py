import torch
from torch.utils.data import DataLoader
from typing import Tuple


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
    # Call propensityScore to use the propensity for each input feature vector