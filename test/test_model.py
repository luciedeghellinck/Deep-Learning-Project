import torch as th

from src.train.model import CATEModel


class TestCATEModel:
    def test_hypothesis(self):
        model = CATEModel(
            input_size=25,
            dim_hidden_layers=100,
            dim_representation=1,
            n_hidden_layers=3,
            alpha=0.05,
        )
        x = th.rand(64, 25)

        t_0 = th.zeros(64).int()
        t_1 = th.ones(64).int()

        h_0 = model.eval().get_hypothesis(x, t_0)
        h_1 = model.eval().get_hypothesis(x, t_1)

        assert h_0.size() == h_1.size() == (64, 1)
        assert th.equal(h_1 - h_0, model.eval().forward(x))
