import abc
from abc import ABC
from typing import Tuple

import torch as th
from sklearn.ensemble import GradientBoostingRegressor

from src.train.model import CATEModel


class SelectionMetric(ABC):
    def __init__(self, models):
        self.models = models
        self.rating = None

    def get_best_model(self):
        self.rate_models()
        best = th.argmin(self.rating)
        return self.models[best.item()]

    def get_model_ranking(self):
        self.rate_models()
        sorted_indices = th.argsort(self.rating)
        ranking = th.Tensor(
            [
                (sorted_indices == i).nonzero(as_tuple=True)[0]
                for i, _ in enumerate(self.models)
            ]
        )
        return ranking

    def rate_models(self):
        if self.rating is None:
            self.create_rating()
        return self.rating

    @abc.abstractmethod
    def create_rating(self):
        pass


class IPW(SelectionMetric):
    def __init__(self, models, validation_dataset, propensity_regressor):
        super().__init__(models)
        self.dataset = validation_dataset
        self.regressor = propensity_regressor

    def create_rating(self):
        self.rating = []
        for tau in self.models:
            self.rating.append(self.IPW(tau))
        self.rating = th.stack(self.rating)

    def IPW(
        self,
        tau,
    ):
        X, T, Y = self.dataset
        propensity_scores = self.regressor.forward(X)

        plug_in_value = T / propensity_scores * Y - (
            ((1 - T) / (1 - propensity_scores)) * Y
        )
        loss = th.nn.MSELoss(reduction="mean")
        return loss(plug_in_value, tau.predict(X))


class TauRisk(SelectionMetric):
    def __init__(self, models, train_dataset, validation_dataset, propensity_regressor):
        super().__init__(models)
        self.dataset = validation_dataset
        self.regressor = propensity_regressor
        self.outcome_reg = self.outcome_regressor(train_dataset)

    def create_rating(self):
        self.rating = []
        for tau in self.models:
            self.rating.append(self.tau_risk(tau))
        self.rating = th.stack(self.rating)

    def tau_risk(
        self,
        tau,
    ):
        X, T, Y = self.dataset

        propensity_scores = self.regressor.forward(X)
        expected_outcome = self.outcome_reg.predict(X)

        plug_in_value = (
            1 / len(T) * th.sum((Y - expected_outcome) - (T - propensity_scores))
        )
        loss = th.nn.MSELoss(reduction="mean")
        return loss(plug_in_value, tau.predict(X))

    @staticmethod
    def outcome_regressor(dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor]):
        X, _, Y = dataset

        reg = GradientBoostingRegressor()
        reg.fit(X, Y)
        return reg


class CounterfactualCrossValidation(SelectionMetric):
    def __init__(self, models, cate_model: CATEModel, validation_dataset):
        super().__init__(models)
        self.cate_model = cate_model
        self.dataset = validation_dataset

    def create_rating(self):
        self.rating = []
        for tau in self.models:
            self.rating.append(self.counter_factual_cross_validation(tau))
        self.rating = th.stack(self.rating)

    def counter_factual_cross_validation(self, tau):
        loss = th.nn.MSELoss(reduction="mean")
        X, T, Y = self.dataset

        plug_in_value = self.cate_model.forward(X)
        predicted = tau.predict(X)
        return loss(predicted, plug_in_value)
