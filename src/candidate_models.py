from sklearn import tree, ensemble, linear_model, svm
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import torch as th
from typing import Tuple
from sklearn.base import clone


class SLearner:
    def __init__(self, base_learner):
        self.models = [base_learner(), base_learner()]

    def fit(self, data):
        """
        Fit the S-learner on the training data.
        data: tuple of PyTorch tensors
            The training data as a tuple of (X, T, y) tensors.
        """
        X, T, y = data
        X0 = X[T == 0]
        y0 = y[T == 0]
        X1 = X[T == 1]
        y1 = y[T == 1]

        # Fit the models on the treated and untreated groups
        self.models[0].fit((X0, y0))
        self.models[1].fit((X1, y1))

    def predict(self, data):
        """
        Predict the outcome of the S-learner on the input data.
        data: tuple of PyTorch tensors
            The input data as a tuple of (X, T) tensors.
        Returns:
        -------
        y_pred: PyTorch tensor
            The predicted outcome for each input sample.
        """
        X, T = data
        # Predict the outcome for each sample using the appropriate model
        y_pred = torch.zeros(X.shape[0])
        y_pred[T == 0] = self.models[0].predict(X[T == 0])
        y_pred[T == 1] = self.models[1].predict(X[T == 1])
        return y_pred

    def cate(self, X):
        """
        Calculate the conditional average treatment effect (CATE) for the input data.
        X: PyTorch tensor, shape (n_samples, n_features)
            The input samples for which to calculate the CATE.
        Returns:
        -------
        cate: PyTorch tensor, shape (n_samples,)
            The CATE for each input sample.
        """
        y1 = self.models[1].predict(X)
        y0 = self.models[0].predict(X)
        return y1 - y0


def candidatePredictorTau(dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor], algo: string,
                          learner: string) -> th.Tensor:
    if algo == "slearner" and learner == "decisiontree":
        X, T, y = dataset
        train_size = int(0.8 * dataset[0].size()[0])
        X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, y, test_size=0.2, random_state=42)

        slearner = SLearner(base_learner=DecisionTreeRegressor)
        # slearner = Slearner(base_learner=RandomForestRegressor)
        # slearner = Slearner(base_learner=GradientBoostingRegressor)
        # slearner = Slearner(base_learner=Ridge)
        # slearner = Slearner(base_learner=SVR)

        train_data = (X_train, T_train, y_train)
        slearner.fit(train_data)
        # test_data = (X_test, T_test)
        # y_pred = slearner.predict(test_data)
        cate_pred = slearner.cate(X_test)
    else:
        print("error")
