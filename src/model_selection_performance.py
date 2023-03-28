import torch as th
from torch.optim import Adam
from tqdm import tqdm
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

n_hidden_layers = 3
dim_hidden_layers = 100
alpha = 0.356
learning_rate = 4.292 * 10 ** (-4)
batch_size = 256
dropout_rate = 0.2

MLAlgorithms = [DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, Ridge, SVR]
metaLearners = [SLearner, XLearner, TLearner, DomainAdaptationLearner, DRLearner]

dataset = ihdpDataset(csv_file)
input_size = dataset.X.size()[0]
train_ratio = 0.35
validation_ratio = 0.35
test_ratio = 0.30
x_train, x_test, t_train, t_test, y_train, y_test = train_test_split(dataset.X, dataset.T, dataset.Y, test_size=1 - train_ratio)
x_val, x_test, t_val, t_test, y_val, y_test = train_test_split(x_test, t_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))

model = CATEModel(input_size, n_hidden_layers, dim_hidden_layers, alpha)
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = loss

epochs = 100 #They don't give the number of epochs... (this is a random number)

for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train(train_loader, model, optimizer, criterion)
    test_loss, test_acc = test(test_loader, model, criterion)



