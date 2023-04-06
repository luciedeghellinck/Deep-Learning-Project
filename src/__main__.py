import json
from typing import Dict, List
import torch as th

from src.data.data import ihdpDataset
from src.measure.model_comparison_metrics import NRMSE, RankCorrelation, Regret
from src.select.performance_estimation import (IPW,
                                               CounterfactualCrossValidation,
                                               SelectionMetric, TauRisk, PlugIn)
from src.train.candidate_models import CATEModelFactory, ModelFactory
from src.train.propensity_score import PropensityRegressorFactory


def create_selection_methods(dataset_train, dataset_validate, mu_values, propensity_dataset):
    tau_models = ModelFactory().create(dataset_train)
    propensity_regressor = PropensityRegressorFactory(
        propensity_dataset
    ).create()
    CATE_model = CATEModelFactory(propensity_regressor).create(
        dataset_train, dataset_validate
    )
    selection_methods = [IPW(tau_models, dataset_validate, propensity_regressor),
                                   TauRisk(tau_models, dataset_train, dataset_validate, propensity_regressor),
                                   PlugIn(tau_models, dataset_train, mu_values),
                                   CounterfactualCrossValidation(tau_models, CATE_model, dataset_validate)
                                   ]
    return selection_methods


def create_measurements(selection_method, test_data) -> th.Tensor:
    regret = Regret(selection_method, test_data)
    nrmse = NRMSE(selection_method, test_data)
    rank_correlation = RankCorrelation(selection_method, test_data)
    return th.tensor([regret.get_measure(),
                      nrmse.get_measure(),
                      rank_correlation.get_measure()])


#
def printTable1(data):
    data = th.swapaxes(data, 0, 2)  # dim=0: measurement (regret, nrmse, rankCorrelation); dim=1: selection method (IPW, TauRisk, PlugIn, CFCV); dim=2: 100 sets
    table1 = th.empty(3, 4, 3)  # dim=0: 3 metrics; dim=1: 4 methods; dim=2: mean, stdErr, Worst-Case
    for metric in range(3):
        for method in range(4):
            table1[metric][method][0] = th.mean(data[metric][method])
            table1[metric][method][1] = th.std(data[metric][method])
            if method == 2:  # for rankCorrelation
                table1[metric][method][2] = th.min(data[metric][method])
            else:  # for regret and nrmse
                table1[metric][method][2] = th.max(data[metric][method])

    print("IPW:\n")
    print("   Regret:\n")
    print("      Mean: " + table1[0][0][0] + "   Standard deviation: " + table1[0][0][1] + "   Worst case: " + table1[0][0][2])
    print("   NRMSE:\n")
    print("      Mean: " + table1[1][0][0] + "   Standard deviation: " + table1[1][0][1] + "   Worst case: " + table1[1][0][2])
    print("   Rank Correlation:\n")
    print("      Mean: " + table1[2][0][0] + "   Standard deviation: " + table1[2][0][1] + "   Worst case: " + table1[2][0][2])

    print("Tau Risk:\n")
    print("   Regret:\n")
    print("      Mean: " + table1[0][1][0] + "   Standard deviation: " + table1[0][1][1] + "   Worst case: " + table1[0][1][2])
    print("   NRMSE:\n")
    print("      Mean: " + table1[1][1][0] + "   Standard deviation: " + table1[1][1][1] + "   Worst case: " + table1[1][1][2])
    print("   Rank Correlation:\n")
    print("      Mean: " + table1[2][1][0] + "   Standard deviation: " + table1[2][1][1] + "   Worst case: " + table1[2][1][2])

    print("Plug-in:\n")
    print("   Regret:\n")
    print("      Mean: " + table1[0][2][0] + "   Standard deviation: " + table1[0][2][1] + "   Worst case: " + table1[0][2][2])
    print("   NRMSE:\n")
    print("      Mean: " + table1[1][2][0] + "   Standard deviation: " + table1[1][2][1] + "   Worst case: " + table1[1][2][2])
    print("   Rank Correlation:\n")
    print("      Mean: " + table1[2][2][0] + "   Standard deviation: " + table1[2][2][1] + "   Worst case: " + table1[2][2][2])

    print("CFCV:\n")
    print("   Regret:\n")
    print("      Mean: " + table1[0][3][0] + "   Standard deviation: " + table1[0][3][1] + "   Worst case: " + table1[0][3][2])
    print("   NRMSE:\n")
    print("      Mean: " + table1[1][3][0] + "   Standard deviation: " + table1[1][3][1] + "   Worst case: " + table1[1][3][2])
    print("   Rank Correlation:\n")
    print("      Mean: " + table1[2][3][0] + "   Standard deviation: " + table1[2][3][1] + "   Worst case: " + table1[2][3][2])

def main():
    dataset = ihdpDataset("../dataset/ihdp_npci_1-100.test.npz", (0.35, 0.35, 0.30))
    data = th.empty(len(dataset), 4,
                    3)  # dim=0: 100 sets; dim=1: selection method (IPW, TauRisk, PlugIn, CFCV); dim=2: measurement (regret, nrmse, rankCorrelation);
    for i, (dataset_train, dataset_validate, mu_values, dataset_test) in enumerate(dataset):
        selection_methods = create_selection_methods(
            dataset_train, dataset_validate, mu_values, dataset.get_propensity_dataset()
        )  # Tensor: [IPW, TauRisk, PlugIn, CFCV]
        for j, selection_method in enumerate(selection_methods):
            measurements = create_measurements(selection_method,
                                               dataset_test)  # Tensor: [regret, nrmse, rankCorrelation]
            data[i][j] = measurements
    printTable1(data)


if __name__ == "__main__":
    main()
