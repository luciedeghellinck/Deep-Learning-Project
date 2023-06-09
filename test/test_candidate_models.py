import torch as th
from econml.dr import DRLearner
from econml.metalearners import (DomainAdaptationLearner, SLearner, TLearner,
                                 XLearner)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.train.candidate_models import candidatePredictorTau


class TestCandidateModels:
    def test_candidate_models_SLearner(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        xtest = th.rand(10, 25)

        learner = SLearner
        algo = DecisionTreeRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = SLearner
        algo = RandomForestRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = SLearner
        algo = GradientBoostingRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = SLearner
        algo = Ridge()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = SLearner
        algo = SVR()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

    def test_candidate_models_XLearner(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        xtest = th.rand(10, 25)

        learner = XLearner
        algo = DecisionTreeRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = XLearner
        algo = RandomForestRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = XLearner
        algo = GradientBoostingRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = XLearner
        algo = Ridge()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = XLearner
        algo = SVR()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

    def test_candidate_models_TLearner(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        xtest = th.rand(10, 25)

        learner = TLearner
        algo = DecisionTreeRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = TLearner
        algo = RandomForestRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = TLearner
        algo = GradientBoostingRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = TLearner
        algo = Ridge()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = TLearner
        algo = SVR()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

    def test_candidate_models_DomainAdaptationLearner(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        xtest = th.rand(10, 25)

        learner = DomainAdaptationLearner
        algo = DecisionTreeRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DomainAdaptationLearner
        algo = RandomForestRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DomainAdaptationLearner
        algo = GradientBoostingRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DomainAdaptationLearner
        algo = Ridge()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DomainAdaptationLearner
        algo = SVR()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

    def test_candidate_models_DRLearner(self):
        x = th.rand(64, 25)
        t = th.randint(0, 2, (64,))
        y = th.rand(64)
        dataset = (x, t, y)
        xtest = th.rand(10, 25)

        learner = DRLearner
        algo = DecisionTreeRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DRLearner
        algo = RandomForestRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DRLearner
        algo = GradientBoostingRegressor()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DRLearner
        algo = Ridge()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10

        learner = DRLearner
        algo = SVR()
        tau = candidatePredictorTau(dataset, xtest, algo, learner)
        assert tau.size()[0] == 10
