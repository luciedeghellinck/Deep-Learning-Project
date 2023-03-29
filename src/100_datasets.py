import torch as th
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from econml.metalearners import SLearner, XLearner, TLearner, DomainAdaptationLearner
from econml.dr import DRLearner
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from src.data import ihdpDataset
from src.model import CATEModel
from src.loss import loss
from src.training import fit, train, test
from src.candidate_models import candidatePredictorTau
from src.performance_estimation import performanceEstimator, IPW, tau_risk, outcome_regressor
from src.model_comparison_metrics import rankCorrelation, Regret, NRMSE
from src.propensity_score import propensityRegression

n_hidden_layers = 3
dim_hidden_layers = 100
alpha = 0.356
learning_rate = 4.292 * 10 ** (-4)
batch_size = 256
dropout_rate = 0.2
train_ratio = 0.35
validation_ratio = 0.35
test_ratio = 0.30

MLAlgorithms = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), Ridge(), SVR()]
metaLearners = [SLearner, XLearner, TLearner, DomainAdaptationLearner, DRLearner]

#CHECK WHAT TO DO WITH THE TWO DATA FILES
dataset = ihdpDataset("../dataset/ihdp_npci_1-100.test.npz")
realisation_number = dataset.X.size()[0]  # 100 for this dataset
input_size = dataset.X.size()[1]

model = CATEModel(input_size, n_hidden_layers, dim_hidden_layers, 1, alpha, dropout_rate)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = loss

metrics = th.empty(3, 4, realisation_number) # dim=0: metric (Correlation, regret, NRMSE); dim=1: Methods (IPW, TauRisk, PlugIn, CFCV); dim=2: realisations

for i, (X, T, Y) in enumerate(zip(dataset.X, dataset.T, dataset.Y)):
    x_train, x_test, t_train, t_test, y_train, y_test = train_test_split(X.clone().detach(), T.clone().detach(),
                                                                         Y.clone().detach(), test_size=1 - train_ratio)
    x_val, x_test, t_val, t_test, y_val, y_test = train_test_split(x_test, t_test, y_test, test_size=test_ratio / (
            test_ratio + validation_ratio))

    datasetTrain = x_train, t_train, y_train
    loader = DataLoader(TensorDataset(*datasetTrain), batch_size=batch_size)

    # TRAINING SET

    # VALIDATION SET
    ## TauTilde (apply the neural network obtained in the training set to the validation set feature vectors)
    tauTildeVal = th.empty(len(y_val))

    ## TauHat and comparison
    bestPerformanceCFCV, bestPerformanceIPW, bestPerformanceTauRisk, bestPerformancePlugIn = float('inf'), float('inf'), float('inf'), float('inf')
    tauSelectedCFCV, tauSelectedIPW, tauSelectedTauRisk, tauSelectedPlugIn = th.empty(len(y_val)), th.empty(len(y_val)), th.empty(len(y_val)), th.empty(len(y_val))
    for algo in MLAlgorithms:
        for learner in metaLearners:
            tauHat = candidatePredictorTau((x_train, t_train, y_train), x_val, algo, learner)
            ## CHECK THE ARGUMENTS
            performanceCFCV = performanceEstimator(tauTildeVal, tauHat)
            performanceIPW = IPW((x_train, t_train, y_train), propensityRegression, tauHat)
            performanceTauRisk = tau_risk((x_train, t_train, y_train), propensityRegression, tauHat)
            performancePlugIn = outcome_regressor((x_train, t_train, y_train))
            if performanceCFCV < bestPerformanceCFCV:
                bestPerformanceCFCV = performanceCFCV
                tauSelectedCFCV = tauHat
            if performanceIPW < bestPerformanceIPW:
                bestPerformanceIPW = performanceIPW
                tauSelectedIPW = tauHat
            if performanceTauRisk < bestPerformanceTauRisk:
                bestPerformanceTauRisk = performanceTauRisk
                tauSelectedTauRisk = tauHat
            if performancePlugIn < bestPerformancePlugIn:
                bestPerformancePlugIn = performancePlugIn
                tauSelectedPlugIn = tauHat

    # TEST SET
    ## TauTilde (apply the neural network obtained in the training set to the validation set feature vectors)
    tauTildeTest = th.empty(len(y_test))

    ## TauHat and comparison
    bestPerformanceCFCV, bestPerformanceIPW, bestPerformanceTauRisk, bestPerformancePlugIn = float('inf'), float('inf'), float('inf'), float('inf')
    tauBestCFCV, tauBestIPW, tauBestTauRisk, tauBestPlugIn = th.empty(len(y_val)), th.empty(len(y_val)), th.empty(len(y_val)), th.empty(len(y_val))
    for algo in MLAlgorithms:
        for learner in metaLearners:
            tauHat = candidatePredictorTau((x_train, t_train, y_train), x_test, algo, learner)
            performanceCFCV = performanceEstimator(tauTildeTest, tauHat)
            ## CHECK THE ARGUMENTS -> what changes compared to the validation set? Where do we add the tauhat in the outcome regressor...
            performanceIPW = IPW((x_train, t_train, y_train), propensityRegression, tauHat)
            performanceTauRisk = tau_risk((x_train, t_train, y_train), propensityRegression, tauHat)
            performancePlugIn = outcome_regressor((x_train, t_train, y_train))
            if performanceCFCV < bestPerformanceCFCV:
                bestPerformanceCFCV = performanceCFCV
                tauBestCFCV = tauHat
            if performanceIPW < bestPerformanceIPW:
                bestPerformanceIPW = performanceIPW
                tauBestIPW = tauHat
            if performanceTauRisk < bestPerformanceTauRisk:
                bestPerformanceTauRisk = performanceTauRisk
                tauBestTauRisk = tauHat
            if performancePlugIn < bestPerformancePlugIn:
                bestPerformancePlugIn = performancePlugIn
                tauBestPlugIn = tauHat
    ## Evaluate the metrics
    # TAUSELECTED AND TAUBEST DO NOT HAVE THE SAME SIZE SINCE THERE IS A DIFFERENT NUMBER OF INPUT DATA --> HOW ARE WE MEANT TO COMPARE THEM... SINCE THE FEATURES ARE DIFFERENT, IT DOESNT MAKE SENSE
    metrics[0][0][i] = rankCorrelation(tauSelectedIPW, tauBestIPW)
    metrics[0][1][i] = rankCorrelation(tauSelectedTauRisk, tauBestTauRisk)
    metrics[0][2][i] = rankCorrelation(tauSelectedPlugIn, tauBestPlugIn)
    metrics[0][3][i] = rankCorrelation(tauSelectedCFCV, tauBestCFCV)

    metrics[1][0][i] = Regret(tauSelectedIPW, tauBestIPW, tauTildeTest)
    metrics[1][1][i] = Regret(tauSelectedTauRisk, tauBestTauRisk, tauTildeTest)
    metrics[1][2][i] = Regret(tauSelectedPlugIn, tauBestPlugIn, tauTildeTest)
    metrics[1][3][i] = Regret(tauSelectedCFCV, tauBestCFCV, tauTildeTest)

    metrics[2][0][i] = NRMSE(tauSelectedIPW, tauBestIPW)
    metrics[2][1][i] = NRMSE(tauSelectedTauRisk, tauBestTauRisk)
    metrics[2][2][i] = NRMSE(tauSelectedPlugIn, tauBestPlugIn)
    metrics[2][3][i] = NRMSE(tauSelectedCFCV, tauBestCFCV)

table1 = th.empty(4, 3, 3)  # dim=0: 4 methods; dim=1: 3 metrics; dim=2: mean, stdErr, Worst-Case

for method in range(4):
    for metric in range(3):
        table1[method][metric][0] = th.mean(metrics[metric][method])
        table1[method][metric][1] = th.std(metrics[metric][method])
        if method == 0:
            table1[method][metric][2] = th.min(metrics[metric][method])
        else:
            table1[method][metric][2] = th.max(metrics[metric][method])
