import torch as th

from src.propensity_score import propensityRegression


def test_propensity_estimator():
    th.manual_seed(1)
    t_0 = th.zeros(200)
    t_1 = th.ones(200)
    t = th.concat([t_0, t_1], dim=0)
    # linearly seperable so should be reproducable
    x_0 = th.rand(200, 1)
    x_1 = th.rand(200, 1) + 1
    x = th.concat([x_0, x_1], dim=0)

    model = propensityRegression((x, t.int(), th.zeros(1)))

    indices = th.round(model.forward(x).squeeze(dim=-1)).int()
    assert th.all(indices == t)

