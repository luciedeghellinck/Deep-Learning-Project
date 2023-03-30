import pytest
import torch as th
from src.train.loss import (
    loss,
    weight,
    adaptedWeight,
    pi,
    compute_distributional_distance,
)
from src.train.model import CATEModel


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

        assert type(pi_0.item()) == float

    def test_adaptedWeight(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        adaptedWeights = adaptedWeight(dataset)

        assert adaptedWeights.size()[0] == 64

    def test_loss(self):
        model = CATEModel(
            input_size=25,
            dim_hidden_layers=100,
            dim_representation=100,
            n_hidden_layers=3,
            alpha=0.05,
        )

        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)

        l = loss(dataset, model, 0.05)

        assert l.size() == ()

    @pytest.mark.parametrize(
        "representation_dimension, distribution_offset",
        [(1, 1), (25, 1), (1, 2), (25, 2), (1, 3), (25, 3)],
    )
    def test_loss_values(self, representation_dimension, distribution_offset):
        th.random.manual_seed(1)
        t_first = th.zeros(200)
        t_second = th.ones(200)
        x_first = th.rand(200, representation_dimension)
        x_second = th.rand(200, representation_dimension) + distribution_offset

        y = th.rand(400)

        dataset = (
            th.concat([x_first, x_second], dim=0),
            th.concat([t_first, t_second], dim=0).int(),
            y,
        )
        model = TestModel()
        divergence = compute_distributional_distance(dataset, model, 1)

        assert divergence == pytest.approx(
            representation_dimension * distribution_offset**2, rel=0.1
        )


class TestModel(CATEModel):
    def __init__(self):
        super().__init__(1, 1, 1, 1, 1)
        self.head_0 = th.nn.Identity()
        self.head_1 = th.nn.Identity()
        self.phi = th.nn.Identity()
