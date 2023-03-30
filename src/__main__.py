import json
from typing import List, Dict

from src.data.data import ihdpDataset
from src.measure.model_comparison_metrics import Regret, NRMSE, RankCorrelation
from src.select.performance_estimation import (
    IPW,
    TauRisk,
    CounterfactualCrossValidation,
    SelectionMetric,
)
from src.train.candidate_models import ModelFactory, CATEModelFactory


def create_selection_methods(dataset_train, dataset_validate):
    tau_models = ModelFactory().create(dataset_train)
    CATE_model, propensity_regressor = CATEModelFactory().create(
        dataset_train, dataset_validate
    )
    selection_methods = {
        "ipw": IPW(tau_models, dataset_validate, propensity_regressor),
        "taurisk": TauRisk(
            tau_models, dataset_train, dataset_validate, propensity_regressor
        ),
        "counterfactual": CounterfactualCrossValidation(
            tau_models, CATE_model, dataset_validate
        ),
    }

    return selection_methods


def create_measurements(selection_method, test_data, ground_truth_cate):
    regret = Regret(selection_method, test_data, ground_truth_cate)
    nrmse = NRMSE(selection_method, test_data, ground_truth_cate)
    rank_correlation = RankCorrelation(selection_method, test_data, ground_truth_cate)
    return {
        "regret": regret.get_measure(),
        "nrmse": nrmse.get_measure(),
        "rank_correlation": rank_correlation.get_measure(),
    }


def main():
    dataset = ihdpDataset("../dataset/ihdp_npci_1-100.test.npz", (35, 35, 30))
    data = []
    for ground_truth_cate, dataset_train, dataset_validate, dataset_test in dataset:
        selection_methods: Dict[str, SelectionMetric] = create_selection_methods(
            dataset_train, dataset_validate
        )
        data.append(
            {
                key: create_measurements(
                    selection_method, dataset_test, ground_truth_cate
                )
                for key, selection_method in selection_methods
            }
        )
    with open("./measurement_output.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
