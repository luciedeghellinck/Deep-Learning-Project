import json
from typing import Dict, List

from src.data.data import ihdpDataset
from src.measure.model_comparison_metrics import NRMSE, RankCorrelation, Regret
from src.select.performance_estimation import (IPW,
                                               CounterfactualCrossValidation,
                                               SelectionMetric, TauRisk, PlugIn)
from src.train.candidate_models import CATEModelFactory, ModelFactory


def create_selection_methods(dataset_train, dataset_validate, mu_values):
    tau_models = ModelFactory().create(dataset_train)
    CATE_model, propensity_regressor = CATEModelFactory().create(
        dataset_train, dataset_validate
    )
    selection_methods = {
        "ipw": IPW(tau_models, dataset_validate, propensity_regressor),
        "taurisk": TauRisk(
            tau_models, dataset_train, dataset_validate, propensity_regressor
        ),
        "plug_in": PlugIn(
            tau_models, dataset_train, mu_values
        ),
        "counterfactual": CounterfactualCrossValidation(
            tau_models, CATE_model, dataset_validate
        ),
    }

    return selection_methods


def create_measurements(selection_method, test_data):
    regret = Regret(selection_method, test_data)
    nrmse = NRMSE(selection_method, test_data)
    rank_correlation = RankCorrelation(selection_method, test_data)
    return {
        "regret": regret.get_measure(),
        "nrmse": nrmse.get_measure(),
        "rank_correlation": rank_correlation.get_measure(),
    }


def main():
    dataset = ihdpDataset("../dataset/ihdp_npci_1-100.train.npz", (0.35, 0.35, 0.30))
    data = []
    for dataset_train, dataset_validate, mu_values, dataset_test in dataset:
        selection_methods: Dict[str, SelectionMetric] = create_selection_methods(
            dataset_train, dataset_validate, mu_values
        )
        data.append(
            {
                key: create_measurements(selection_method, dataset_test)
                for key, selection_method in selection_methods.items()
            }
        )
    with open("./measurement_output.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
