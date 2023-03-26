import pytest
import torch as th
from src.loss import loss, weight, adaptedWeight, pi
from src.model import CATEModel


class TestLoss:

    def test_weight(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        weights = weight(dataset)

        assert weights.size()[0] == 64

    def test_pi(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        pi_0 = pi(dataset, 0)

        # This assert doesn't work although l has a size of 1
        assert pi_0.size() == 1

    def test_adaptedWeight(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        adaptedWeights = adaptedWeight(dataset)

        assert adaptedWeights.size()[0] == 64

    def test_loss(self):
        model = CATEModel(input_size=25, dim_hidden_layers=100, dim_representation=1, n_hidden_layers=3, alpha=0.05)

        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)

        l = loss(dataset, model, 0.05)

        #This assert doesn't work although l has a size of 1
        assert l.size() == 1







