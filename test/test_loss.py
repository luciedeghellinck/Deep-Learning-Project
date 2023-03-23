import pytest
import torch as th
from src.loss import IPM
from src.model import CATEModel


class TestLoss:

    def test_IPM(self):
        # def IPM(dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        X = th.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        T = th.Tensor([0, 1, 0])
        Y = th.Tensor([0.1, 0.5, 1.6])
        model = CATEModel(input_size=3, dim_hidden_layers=100, dim_representation=1, n_hidden_layers=3, alpha=0.05)

        ipm = IPM((X, T, Y), model)

        assert ipm.size() == (1,)
