from econml.metalearners import SLearner, XLearner, TLearner, DomainAdaptationLearner
from econml.dr import DRLearner
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import torch as th
from typing import Tuple
import numpy as np


def candidatePredictorTau(dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], Xtest: th.tensor, algo, learner) -> th.Tensor:
    if learner == SLearner or learner == XLearner or learner == TLearner:
        est = SLearner(overall_model=algo)
    elif learner == DomainAdaptationLearner:
        est = learner(models=algo, final_models=algo)
    elif learner == DRLearner:
        est = learner()
    else:
        print("This meta learner is not accepted")

    est.fit(dataset[2].numpy(), dataset[1].numpy(), X=dataset[0].numpy())
    effect = est.effect(Xtest.numpy())
    return th.from_numpy(effect)

# def performanceEstimator(tau: th.Tensor, candidate: th.Tensor):

