import torch as th

def plugInTau(x, t, y, f0, f1, regression):
    """
    Calculates the plug in tau for the feature vector.

    Args:
      x: m dimensional float feature vector.
      t: 0/1 treatment type.
      y: float outcome.
      f0: hypothesis function evaluated at the feature vector when there is no treatment.
      f1: hypothesis function evaluated at the feature vector when there is a treatment.
      regression: regression function for the propensity.

    Returns:
      A float representing the plug-in predictor for the datapoint [X,T,Y] given the hypotheses functions f0 and f1 as well as the propensity regression function.
    """


def candidatePredictorTau(x, MLAlgorithm, metalearner):
    """
    Calculates the tau for a given a Machine Learning algorithm, a meta-learner and a feature vector.

    Args:
      x: m dimensional float feature vector.
      MLAlgorithm: Machine-learning algorithm.
      metaLearner: metal-learner.
    Returns:
      A float representing the CATE predictor for the algorithm and the meta-learner for the feature.
    """
    # If-else function that checks the machine learning algorithm and the metalearner type
    pass


def performanceEstimator(plugIn, candidate, features: th.Tensor):
    """
    Calculates the performance estimator for a set of plug-in and candidate tau.

    Args:
      plugIn: float plug-in CATE preictor
      candidate: float candidate CATE predictor from the Maching learning algorithms and the meta-learners
    Returns:
      A float representing the performance estimator between two CATE predictors.
    """
    # Equation 5
    pass