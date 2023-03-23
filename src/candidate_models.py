from sklearn import tree, ensemble, linear_model, svm
import pytorch as th

def candidatePredictorTau(features: th.Tensor, algo: string, learner: string) -> th.Tensor:
    # MLAlgorithms = [decisionTree, randomForest, gradientBoostingTree, ridgeRegressor, supportVectorRergessor]
    # metaLearners = [sLearner, xLearner, tLearner, domainAdaptationLearner, doublyRobustLearner]
    if algo == 'decisionTree':

