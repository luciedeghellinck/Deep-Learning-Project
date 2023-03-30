import torch as th

from src.measure.model_comparison_metrics import (NRMSE, Regret, Rtrue,
                                                  rankCorrelation)


class TestComparisonMetrics:
    def test_Rtrue(self):
        tau_hat = th.rand(64)
        tau_tilde = th.rand(64)

        rtrue = Rtrue(tau_hat, tau_tilde)
        assert type(rtrue) == float

    def test_Regret(self):
        tau_hat_selected = th.rand(64)
        tau_hat_best = th.rand(64)
        tau_tilde = th.rand(64)

        regret = Regret(tau_hat_selected, tau_hat_best, tau_tilde)
        assert type(regret) == float

    def test_rankCorrelation(self):
        tau_hat_selected = th.rand(64)
        tau_hat_best = th.rand(64)

        spearman = rankCorrelation(tau_hat_selected, tau_hat_best)
        assert type(spearman) == float

    def test_nrmse(self):
        tau_hat_selected = th.rand(64)
        tau_hat_best = th.rand(64)

        nrmse = NRMSE(tau_hat_selected, tau_hat_best)
        assert type(nrmse) == float
