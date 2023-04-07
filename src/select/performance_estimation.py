import abc
import logging
from abc import ABC
from typing import List, Tuple


import torch as th
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor

from src.train.model import CATEModel


class SelectionMetric(ABC):
    def __init__(self, models):
        self.models = models
        self.rating = None

    def get_best_model(self):
        self.rate_models()
        best = th.argmin(self.rating)
        logging.debug(f"Best model for {self.__class__.__name__} was  {best.item()}")
        return self.models[best.item()]

    def get_model_ranking(self):
        self.rate_models()
        sorted_indices = th.argsort(self.rating)
        ranking = [
                (sorted_indices == i).nonzero(as_tuple=True)[0].item()
                for i, _ in enumerate(self.models)
            ]
        logging.debug(f"model ranking for {self.__class__.__name__} was {ranking}")
        return th.Tensor(ranking)

    def rate_models(self):
        if self.rating is None:
            self.create_rating()
        return self.rating

    @abc.abstractmethod
    def create_rating(self):
        pass


class IPW(SelectionMetric):
    def __init__(self, models: List[RegressorMixin], validation_dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], propensity_regressor: th.nn.Module):
        super().__init__(models)
        self.val_dataset = validation_dataset
        self.regressor = propensity_regressor

    def create_rating(self):
        self.rating = []
        for tau_hat in self.models:
            self.rating.append(self.IPW(tau_hat))
        self.rating = th.stack(self.rating)

    def IPW(self, tau_hat):
        X, T, Y = self.val_dataset
        propensity_scores = self.regressor.forward(X).squeeze()

        plug_in_value = T / propensity_scores * Y - (
            ((1 - T) / (1 - propensity_scores)) * Y
        )
        loss = th.nn.MSELoss(reduction="mean")
        return loss(plug_in_value, th.from_numpy(tau_hat.effect(X.numpy())))


class TauRisk(SelectionMetric):
    def __init__(self, models: List[RegressorMixin], train_dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], validation_dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], propensity_regressor: th.nn.Module):
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

        propensity_scores = self.regressor.forward(X).squeeze()
        expected_outcome = self.outcome_reg.predict(X)

        plug_in_value = (
            (Y - expected_outcome) - (T - propensity_scores)
        )
        loss = th.nn.MSELoss(reduction="mean")
        return loss(plug_in_value, th.from_numpy(tau.effect(X)))

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

        plug_in_value = self.cate_model.forward(X).squeeze()
        predicted = th.from_numpy(tau.effect(X))
        return loss(predicted, plug_in_value)

class PlugIn(SelectionMetric):
    def __init__(self, models,
                 validation_dataset,
                 mu_values):
        super().__init__(models)
        self.dataset = validation_dataset
        self.mu_values = mu_values

    def create_rating(self):
        self.rating = []
        for tau in self.models:
            self.rating.append(self.plug_in_validation(tau))
        self.rating = th.stack(self.rating)

    def plug_in_validation(self, tau):
        loss = th.nn.MSELoss(reduction="mean")
        X, T, Y = self.dataset
        mu0, mu1 = self.mu_values

        plug_in_value = mu1 - mu0
        predicted = th.from_numpy(tau.effect(X))
        return loss(predicted, plug_in_value)
