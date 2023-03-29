from econml.metalearners import SLearner, XLearner, TLearner, DomainAdaptationLearner
from econml.dr import DRLearner
import torch as th
from typing import Tuple

def candidatePredictorTau(dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], Xtest: th.tensor, algo, learner) -> th.Tensor:
    if learner == SLearner:
        est = learner(overall_model=algo)
    elif learner == XLearner or learner == TLearner:
        est = learner(models=algo)
    elif learner == DomainAdaptationLearner:
        est = learner(models=algo, final_models=algo)
    elif learner == DRLearner:
        est = learner()
    else:
        print("This meta learner is not accepted")

    est.fit(dataset[2].numpy(), dataset[1].numpy(), X=dataset[0].numpy())
    effect = est.effect(Xtest.numpy())
    return th.from_numpy(effect)

