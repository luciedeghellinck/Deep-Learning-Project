import torch as th
import torch.nn as nn
from typing import Tuple
from src.train.model import CATEModel


class SinkhornDistance(nn.Module):
    """
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction="none"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (
            th.empty(batch_size, x_points, dtype=th.float, requires_grad=False)
            .fill_(1.0 / x_points)
            .squeeze()
        )
        nu = (
            th.empty(batch_size, y_points, dtype=th.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .squeeze()
        )

        u = th.zeros_like(mu)
        v = th.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps * (th.log(mu + 1e-8) - th.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    th.log(nu + 1e-8)
                    - th.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = th.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = th.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = th.sum((th.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


# def IPM(dataset: Tuple[th.Tensor, th.Tensor, th.Tensor], model: CATEModel) -> th.Tensor:
#     """
#   Calculates the IPM distance for two probability functions.
#
#   Args:
#     dataset: torch tensor where each row represents a person, the first
#       column represents the float feature vector, the second is the 0/1
#       treatment type and the third is the float outcome.
#     prediction_treated:
#     prediction_untreated:
#   Returns:
#     The IPM distance evaluated on all
#   """
#     with th.no_grad():
#         indices_not_treated = th.nonzero(dataset[1] == 0)
#         indices_treated = th.nonzero(dataset[1] == 1)
#
#         x_not_treated = dataset[0][indices_not_treated].squeeze(dim=1)
#         x_treated = dataset[0][indices_treated].squeeze(dim=1)
#
#         representation_no_treatment = model.get_representation(x_not_treated)
#         representation_treatment = model.get_representation(x_treated)
#
#         ipm = wasserstein_distance(representation_no_treatment.squeeze(dim=-1),
#                                    representation_treatment.squeeze(dim=-1))
#
#     return th.Tensor([ipm])


class Loss(th.nn.Module):
    def __init__(
        self,
        dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor],
        regressor: th.nn.Module,
        alpha: float,
    ):
        super().__init__()
        self.regressor = regressor
        self.alpha = alpha
        self.pi_0 = self.pi(dataset, 0)
        self.pi_1 = self.pi(dataset, 1)
        self.sinkhorn = SinkhornDistance(0.05, 100)
        self.loss_function = th.nn.MSELoss(reduction="none")

    def forward(self, dataset, model):
        return self.loss(dataset, model, self.alpha)

    def loss(
        self,
        dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor],
        model: CATEModel,
        alpha: float,
    ) -> th.Tensor:
        """
        Calculates the new loss function as defined in equation 12

        Args:
          dataset: torch tensor where each row represents a person, the first
            column represents the float feature vector, the second is the 0/1
            treatment type and the third is the float outcome.
          model: CATE Model
          alpha: trade-off hyperparameter

        Returns:
          float loss as defined in equation 12
        """
        empirical_weighted_risk = self.empirical_weighted_risk(dataset, model)
        print(f"weighted risk: {empirical_weighted_risk}")
        distributional_distance = self.distributional_distance(dataset, model, alpha)
        print(f"distributional_distance: {distributional_distance}")
        total_loss = empirical_weighted_risk + distributional_distance
        print(f"total_loss: {total_loss}")
        return total_loss

    def empirical_weighted_risk(
        self,
        dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor],
        model: CATEModel,
    ):
        adapted_weights = self.adaptedWeight(dataset)
        l = self.loss_function(model.get_hypothesis(dataset[0], dataset[1]), dataset[2])

        return th.dot(adapted_weights, l) / dataset[0].size(0)

    def distributional_distance(
        self,
        dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor],
        model: CATEModel,
        alpha: float,
    ):
        sinkhorn = SinkhornDistance(
            self.sinkhorn_generalization, self.sinkhorn_iterations
        )
        indices_not_treated = th.nonzero(dataset[1] == 0)
        indices_treated = th.nonzero(dataset[1] == 1)

        x_not_treated = dataset[0][indices_not_treated].squeeze(dim=1)
        x_treated = dataset[0][indices_treated].squeeze(dim=1)

        representation_no_treatment = model.get_representation(x_not_treated)
        representation_treatment = model.get_representation(x_treated)
        cost, pi, C = sinkhorn.forward(
            representation_no_treatment, representation_treatment
        )
        return alpha * th.sum(cost)

    def weight(self, dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor]) -> th.Tensor:
        """
        Calculates the weight for a given feature vector and a given treatment type.

        Args:
         dataset: torch tensor where each row represents a person, the first
            column represents the float feature vector, the second is the 0/1
            treatment type and the third is the float outcome.
          x: m dimensional float feature vector
          t: 0/1 treatment type.
        Returns:
          The float weight for the feature vector and the treatment type.
        """
        propensity = self.regressor.forward(dataset[0]).squeeze(-1) + 1e-7
        return (dataset[1] * (1 - 2 * propensity) + propensity**2) / (
            propensity * (1 - propensity)
        )

    @staticmethod
    def pi(dataset: Tuple[th.Tensor, th.Tensor, th.Tensor], t: int) -> th.Tensor:
        """
        Calculates the percentage of a given treatment type for the dataset.

        Args:
          dataset: torch tensor where each row represents a person, the first
            column represents the float feature vector, the second is the 0/1
            treatment type and the third is the float outcome.
          t: 0/1 treatment type.

        Returns:
          The float percentage of the treatment type.
        """
        return th.sum(dataset[1] == t) / dataset[1].size()[0]

    def adaptedWeight(
        self, dataset: Tuple[th.Tensor, th.IntTensor, th.Tensor]
    ) -> th.Tensor:
        """
        Calculates the weight for a given feature vector and a given treatment type.

        Args:
         dataset: torch tensor where each row represents a person, the first
            column represents the float feature vector, the second is the 0/1
            treatment type and the third is the float outcome.
          t: 0/1 treatment type.

        Returns:
          The float adapted weight for the feature vector and the treatment type.
        """
        old_weight = self.weight(dataset)
        print(f"old_weight: {old_weight}")
        adapted_weight = th.mul(
            old_weight, 1 / 2 * (dataset[1] / self.pi_1 + (1 - dataset[1]) / self.pi_0)
        )

        return adapted_weight
