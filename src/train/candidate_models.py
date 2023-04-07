from typing import List, Tuple

import torch as th
from econml.dr import DRLearner
from econml.metalearners import (DomainAdaptationLearner, SLearner, TLearner,
                                 XLearner)
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.train.loss import Loss
from src.train.model import CATEModel
from src.train.propensity_score import propensityRegression
from src.train.training import fit


class ModelFactory:
    """Takes in a train dataset and returns 25 models for predicting the CATE"""

    def __init__(self):
        self.learners = [
            SLearner,
            XLearner,
            TLearner,
            DomainAdaptationLearner,
            DRLearner,
        ]
        self.algorithms = [
            DecisionTreeRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
            Ridge,
            SVR,
        ]

    def create(
        self, dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor]
    ) -> List[RegressorMixin]:
        models = []
        for learner in self.learners:
            for algorithm in self.algorithms:
                model = self.candidatePredictorTau(dataset, algorithm(), learner)
                models.append(model)
        return models

    def candidatePredictorTau(
        self, dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], algo, learner
    ) -> RegressorMixin:
        X, T, Y = dataset
        if learner == SLearner:
            est = learner(overall_model=algo)
        elif learner == XLearner or learner == TLearner:
            est = learner(models=algo)
        elif learner == DomainAdaptationLearner:
            est = learner(models=algo, final_models=algo)
        elif learner == DRLearner:
            est = learner(model_regression=algo)
        else:
            raise Exception("This meta learner is not accepted")
        est.fit(Y=Y, T=T, X=X)
        return est


class CATEModelFactory:
    """Takes in a train dataset and propensity regressor and returns a trained CATEModel"""

    def __init__(
        self,
        regressor,
        *,
        n_hidden_layers=3,
        dim_hidden_layers=100,
        input_size=25,
        dim_representation=25,
        batch_size=256,
        alpha=0.356,
        learning_rate=4.292e-4,
        dropout_rate=0.2,
        epochs=30
    ):
        self.regressor = regressor
        self.model_param_dict = {
            "input_size": input_size,
            "n_hidden_layers": n_hidden_layers,
            "dim_hidden_layers": dim_hidden_layers,
            "dim_representation": dim_representation,
            "dropout_rate": dropout_rate,
        }
        self.batch_size = batch_size
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs

    def create(self, training_dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], validation_dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor]):
        model = CATEModel(**self.model_param_dict)
        loss = Loss(training_dataset, self.regressor, self.alpha)
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        data_loader = DataLoader(TensorDataset(*training_dataset), batch_size=self.batch_size)
        validation_data_loader = DataLoader(
            TensorDataset(*validation_dataset), batch_size=self.batch_size
        )

        fit(data_loader, validation_data_loader, model, optimizer, loss, self.epochs)

        return model
