import pytest
import torch as th
from src.model import CATEModel


class TestCATEModel:

    def test_hypothesis(self):
        model = CATEModel(input_size=25, dim_hidden_layers=100, n_hidden_layers=3, alpha=0.05)
        x = th.rand(64, 25)

        t_0 = th.zeros(64, dtype=th.int64)
        t_1 = th.ones(64, dtype=th.int64)

        h_0 = model.get_hypothesis(x, t_0)
        h_1 = model.get_hypothesis(x, t_1)

        assert th.equal(h_1 - h_0, model.forward(x))
