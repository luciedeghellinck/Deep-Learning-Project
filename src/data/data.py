from collections import Generator
from typing import Tuple

import torch as th
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


class ihdpDataset(TensorDataset):
    def __init__(self, data_path, ratio: Tuple[int, int, int]):
        """Dataset initializer for ihdpDataset.

        :param data_path: The path to the data file.
        :param ratio: tuple with traint test split ratio. first index is train, second validate, third test.
        """
        self.ratio = ratio
        data = np.load(data_path)

        super().__init__(
            th.from_numpy(data["x"]),
            th.from_numpy(data["t"]).int(),
            th.from_numpy(data["yf"]),
        )

    def __iter__(
        self,
    ) -> Generator[
        Tuple[
            Tuple[th.Tensor, th.IntTensor, th.Tensor],
            Tuple[th.Tensor, th.IntTensor, th.Tensor],
            Tuple[th.Tensor, th.IntTensor, th.Tensor],
        ],
        None,
        None,
    ]:
        for X, T, Y in self.tensors:
            x_train, x_test, t_train, t_test, y_train, y_test = train_test_split(
                X.clone().detach(),
                T.clone().detach(),
                Y.clone().detach(),
                test_size=1 - self.ratio[0],
            )
            x_val, x_test, t_val, t_test, y_val, y_test = train_test_split(
                x_test,
                t_test,
                y_test,
                test_size=self.ratio[2] / (self.ratio[2] + self.ratio[1]),
            )
            yield (x_train, t_train, y_train), (x_val, y_val, t_val), (
                x_test,
                y_test,
                t_test,
            )
