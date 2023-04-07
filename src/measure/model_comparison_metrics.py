import abc
import logging
from abc import ABC
from typing import List

import torch as th
from scipy.stats import spearmanr

from src.select.performance_estimation import SelectionMetric


class Measurement(SelectionMetric, ABC):
    def __init__(self, selection_method: SelectionMetric, test_dataset):
        super().__init__(selection_method.models)
        self.selection_method = selection_method
        self.dataset = test_dataset[:-1]
        self.test_tau_values = test_dataset[-1]

    def create_rating(self):
        self.rating = []
        for tau in self.models:
            self.rating.append(self.r_true(tau))
        self.rating = th.stack(self.rating)

    @staticmethod
    def log_measure(name):
        def top_level_wrapper(measure_func):
            def wrapper(self):
                outcome = measure_func(self)
                logging.info(f"outcome of {name} was {outcome}")
                return outcome
            return wrapper
        return top_level_wrapper

    @abc.abstractmethod
    def get_measure(self):
        pass

    def r_true(self, model):
        """
        Calculates the performance metric R_true as defined in Equation 1

        tauHat: 1D tensor of size n containing the CATE prediction using a specific ML algorithm and meta-learner
        tauTilde: 1D tensor of size n containing the prediction using the plug in tau
        Returns:
            The performance metric as defined in Equation 1
        """
        loss = th.nn.MSELoss(reduction="sum")
        X, T, Y = self.dataset
        tauHat = th.from_numpy(model.effect(X))
        tauTilde = self.test_tau_values
        return loss(tauHat, tauTilde)


class Regret(Measurement):
    @Measurement.log_measure("regret")
    def get_measure(self):
        """
        Calculates the regret: the difference between the true performance of the selected model and that of the best
        possible candidate in M

        Args:
            tauHatSelected: 1D tensor of size n containing the best CATE prediction calculated using the validation set
            tauHatBest: 1D tensor of size n containing the best CATE prediction calculated using the test set
            tauTilde: 1D tensor of size n containing the best CATE prediction calculated using the neural network
        Returns:
            The regret between the two CATE predictors
        """
        r_selected = self.r_true(self.selection_method.get_best_model())
        r_best = self.r_true(self.get_best_model())
        return th.sum((r_selected - r_best) / r_best)


class RankCorrelation(Measurement):
    @Measurement.log_measure("RankCorrelation")
    def get_measure(self):
        """
        Calculates the Spearman Rank Correlation

        Returns:
            The Spearman Rank Correlation between the two CATE predictors
        """
        correlation = spearmanr(
            self.get_model_ranking(), self.selection_method.get_model_ranking()
        )
        return float(correlation.pvalue)


class NRMSE(Measurement):
    @Measurement.log_measure("NRMSE")
    def get_measure(self):
        """
        Calculates the normalised root mean sqaured error between the CATE predictors

        Args:
            tauHatSelected: 1D tensor of size n containing the best CATE prediction calculated using the validation set
            tauHatBest: 1D tensor of size n containing the best CATE prediction calculated using the test set
        Returns:
            The NRMSE between the two CATE predictors
        """
        squared_error = self.r_true(self.selection_method.get_best_model())
        nrmse = th.mean(squared_error) / th.var(self.test_tau_values)
        return th.sqrt(nrmse)
