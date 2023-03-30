import torch as th
from econml.dr import DRLearner
from econml.metalearners import (DomainAdaptationLearner, SLearner, TLearner,
                                 XLearner)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.data import ihdpDataset
from src.measure.model_comparison_metrics import NRMSE, Regret, rankCorrelation
from src.select.performance_estimation import (IPW, outcome_regressor,
                                               performanceEstimator, tau_risk)
from src.train.candidate_models import candidatePredictorTau
from src.train.loss import Loss, loss
from src.train.model import CATEModel
from src.train.propensity_score import propensityRegression
from src.train.training import fit

n_hidden_layers = 3
dim_hidden_layers = 100
alpha = 0.356
learning_rate = 4.292 * 10 ** (-4)
batch_size = 256
dropout_rate = 0.2
train_ratio = 0.35
validation_ratio = 0.35
test_ratio = 0.30

MLAlgorithms = [
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    Ridge(),
    SVR(),
]
metaLearners = [SLearner, XLearner, TLearner, DomainAdaptationLearner, DRLearner]

# CHECK WHAT TO DO WITH THE TWO DATA FILES
dataset = ihdpDataset("../dataset/ihdp_npci_1-100.test.npz")
realisation_number = dataset.X.size()[0]  # 100 for this dataset
input_size = 25  # 25 for this dataset

model = CATEModel(25, n_hidden_layers, dim_hidden_layers, 1, alpha, dropout_rate)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = loss

metrics = th.empty(
    4, 3, realisation_number
)  # dim=0: Methods (IPW, TauRisk, PlugIn, CFCV); dim=1: metric (Correlation, regret, NRMSE); dim=2: realisations
table1 = th.empty(
    4, 3, 3
)  # dim=0: 4 methods; dim=1: 3 metrics; dim=2: mean, stdErr, Worst-Case

for i, (X, T, Y) in enumerate(zip(dataset.X, dataset.T, dataset.Y)):
    x_train, x_test, t_train, t_test, y_train, y_test = train_test_split(
        X.clone().detach(),
        T.clone().detach(),
        Y.clone().detach(),
        test_size=1 - train_ratio,
    )
    x_val, x_test, t_val, t_test, y_val, y_test = train_test_split(
        x_test, t_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio)
    )

    datasetTrain = x_train, t_train, y_train
    loader = DataLoader(TensorDataset(*datasetTrain), batch_size=batch_size)

    testLoader = DataLoader(TensorDataset(x_val, t_val, y_val), batch_size=batch_size)
    regressor = propensityRegression(datasetTrain)
    loss = Loss(regressor, alpha)
    # TRAINING SET

    # VALIDATION SET
    ## TauTilde (apply the neural network obtained in the training set to the validation set feature vectors)
    fit(loader, testLoader, model, optimizer, loss, 100)
    tauTildeVal = model.forward(x_val)

    ## TauHat and comparison
    bestPerformance = [
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    ]  # bestPerformanceCFCV, bestPerformanceIPW, bestPerformanceTauRisk, bestPerformancePlugIn
    tauSelected = [
        th.empty(len(y_val)),
        th.empty(len(y_val)),
        th.empty(len(y_val)),
        th.empty(len(y_val)),
    ]  # tauSelectedCFCV, tauSelectedIPW, tauSelectedTauRisk, tauSelectedPlugIn
    for algo in MLAlgorithms:
        for learner in metaLearners:
            tauHat = candidatePredictorTau(
                (x_train, t_train, y_train), x_val, algo, learner
            )
            ## CHECK THE ARGUMENTS
            performanceCFCV = performanceEstimator(tauTildeVal, tauHat)
            performanceIPW = IPW(
                (x_train, t_train, y_train), propensityRegression(datasetTrain), tauHat
            )
            performanceTauRisk = tau_risk(
                (x_train, t_train, y_train), propensityRegression(datasetTrain), tauHat
            )
            performancePlugIn = outcome_regressor((x_train, t_train, y_train))
            if performanceCFCV < bestPerformance[0]:
                bestPerformance[0] = performanceCFCV
                tauSelected[0] = tauHat
            if performanceIPW < bestPerformance[1]:
                bestPerformance[1] = performanceIPW
                tauSelected[1] = tauHat
            if performanceTauRisk < bestPerformance[2]:
                bestPerformance[2] = performanceTauRisk
                tauSelected[2] = tauHat
            if performancePlugIn < bestPerformance[3]:
                bestPerformance[3] = performancePlugIn
                tauSelected[3] = tauHat

    # TEST SET
    ## TauTilde (apply the neural network obtained in the training set to the validation set feature vectors)
    tauTildeTest = th.empty(len(y_test))

    ## TauHat and comparison
    bestPerformance = [
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    ]  # bestPerformanceCFCV, bestPerformanceIPW, bestPerformanceTauRisk, bestPerformancePlugIn
    tauBest = [
        th.empty(len(y_test)),
        th.empty(len(y_test)),
        th.empty(len(y_test)),
        th.empty(len(y_test)),
    ]  # tauBestCFCV, tauBestIPW, tauBestTauRisk, tauBestPlugIn
    for algo in MLAlgorithms:
        for learner in metaLearners:
            tauHat = candidatePredictorTau(
                (x_train, t_train, y_train), x_test, algo, learner
            )
            performanceCFCV = performanceEstimator(tauTildeTest, tauHat)
            ## CHECK THE ARGUMENTS -> what changes compared to the validation set? Where do we add the tauhat in the outcome regressor...
            performanceIPW = IPW(
                (x_train, t_train, y_train), propensityRegression, tauHat
            )
            performanceTauRisk = tau_risk(
                (x_train, t_train, y_train), propensityRegression, tauHat
            )
            performancePlugIn = outcome_regressor((x_train, t_train, y_train))
            if performanceCFCV < bestPerformance[0]:
                bestPerformance[0] = performanceCFCV
                tauBest[0] = tauHat
            if performanceIPW < bestPerformance[1]:
                bestPerformance[1] = performanceIPW
                tauBest[1] = tauHat
            if performanceTauRisk < bestPerformance[2]:
                bestPerformance[2] = performanceTauRisk
                tauBest[2] = tauHat
            if performancePlugIn < bestPerformance[3]:
                bestPerformance[3] = performancePlugIn
                tauBest[3] = tauHat
    ## Evaluate the metrics
    # TAUSELECTED AND TAUBEST DO NOT HAVE THE SAME SIZE SINCE THERE IS A DIFFERENT NUMBER OF INPUT DATA --> HOW ARE WE MEANT TO COMPARE THEM... SINCE THE FEATURES ARE DIFFERENT, IT DOESNT MAKE SENSE

    # metrics: dim=0: Methods (IPW, TauRisk, PlugIn, CFCV); dim=1: metric (Correlation, regret, NRMSE); dim=2: realisations
    for method in range(4):
        metrics[method][0][i] = rankCorrelation(tauSelected[method], tauBest[method])
        metrics[method][1][i] = Regret(
            tauSelected[method], tauBest[method], tauTildeTest
        )
        metrics[method][2][i] = NRMSE(tauSelected[method], tauBest[method])

# table1: dim=0: 4 methods; dim=1: 3 metrics; dim=2: mean, stdErr, Worst-Case
for method in range(4):
    for metric in range(3):
        table1[method][metric][0] = th.mean(metrics[method][metric])
        table1[method][metric][1] = th.std(metrics[method][metric])
        if method == 0:
            table1[method][metric][2] = th.min(metrics[method][metric])
        else:
            table1[method][metric][2] = th.max(metrics[method][metric])
