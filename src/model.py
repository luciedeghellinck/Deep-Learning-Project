import torch as th


class CATEModel(th.nn.Module):

    def __init__(self,
                 input_size: int,
                 n_hidden_layers: int,
                 dim_hidden_layers: int,
                 alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        # initialize representation network phi first
        self.phi = self.__initialize_phi(input_size, n_hidden_layers, dim_hidden_layers)
        # Then initialize the two seperate heads indexed at 0 and 1 (easy t indexing)
        self.heads = (self.__initialize_head(n_hidden_layers, dim_hidden_layers),
                      self.__initialize_head(n_hidden_layers, dim_hidden_layers))

    @staticmethod
    def __initialize_phi(input_size: int,
                         n_hidden_layers: int,
                         dim_hidden_layers: int) -> th.nn.Module:
        # In the original paper it is not actually specified what the neural network architecture looks like
        # For now I will be going for an architecture of size n_hidden_layers, dim_hidden_layers with ReLU inbetween
        layers = [
            th.nn.Linear(in_features=input_size, out_features=dim_hidden_layers),
            th.nn.ReLU(),
        ]
        for _ in range(n_hidden_layers):
            layers += [
                th.nn.Linear(in_features=dim_hidden_layers, out_features=dim_hidden_layers),
                th.nn.ReLU()
            ]
        return th.nn.Sequential(*layers)

    @staticmethod
    def __initialize_head(n_hidden_layers: int,
                          dim_hidden_layers: int) -> th.nn.Module:
        # In the original paper it is not actually specified what the neural network architecture looks like
        # For now I will be going for an architecture of size n_hidden_layers, dim_hidden_layers with ReLU inbetween
        layers = []

        for _ in range(n_hidden_layers):
            layers += [
                th.nn.Linear(in_features=dim_hidden_layers, out_features=dim_hidden_layers),
                th.nn.ReLU()
            ]
        # single output value for h_t
        layers += [
            th.nn.Linear(in_features=dim_hidden_layers, out_features=1)
        ]

        return th.nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Forward in the case of only an x value, means getting the tau for a given example.
        representation = self.get_representation(x)
        return self.heads[1].forward(representation) - self.heads[0].forward(representation)

    def get_representation(self, x: th.Tensor) -> th.Tensor:
        assert len(x.size()) == 2, "x should be a 2 dimensional vector with batch_size on the first axis and features" \
                                   " on the second axis! "
        return self.phi.forward(x)

    def get_hypothesis(self, x: th.Tensor, t: th.IntTensor) -> th.Tensor:
        # get hypothesis, returns the expected value given treatment vector t.
        # t should contain integers {0, 1} and t and x should be aligned along the last dimension of the tensor
        assert len(t.size()) == 1, "t should be a 1 dimensional vector with batch_size on the first axis! was" \
                                   f"{t.size()} instead"
        assert x.size(dim=0) == t.size(dim=0), "x and t should be aligned along their first axis! was " \
                                               f"{x.size(dim=0), t.size(dim=0)} instead."
        representation = self.get_representation(x)
        mask = t==0
        expected_values = mask*self.heads[0].forward(representation).squeeze(-1) + \
                          ~mask*self.heads[1].forward(representation).squeeze(-1)

        return expected_values.unsqueeze(-1)
