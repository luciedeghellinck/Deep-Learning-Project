import torch as th

from src.select.performance_estimation import IPW
from src.train.candidate_models import ModelFactory, CATEModelFactory

class TestPerformanceEstimation:

    def test_IPW(self):
        dataset_train = (th.rand(35, 25), th.randint(2, (35,)).to(th.int32), th.rand(35))
        dataset_validate = (th.rand(35, 25), th.randint(2, (35,)).to(th.int32), th.rand(35))

        tau_hat_models = ModelFactory().create(dataset_train)
        assert len(tau_hat_models) == 25
        CATE_tilde_model, propensity_regressor = CATEModelFactory().create(dataset_train, dataset_validate)
        ipw = IPW(tau_hat_models, dataset_validate, propensity_regressor)
        ipw.IPW(tau_hat_models[0])

        # assert ipw.rating.size()[0] == 25


