from typing import Generator, Tuple

import numpy as np
import torch
import torch as th
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


class ihdpDataset(TensorDataset):
    def __init__(self, data_path, ratio: Tuple[float, float, float]):
        """Dataset initializer for ihdpDataset.

        :param data_path: The path to the data file.
        :param ratio: tuple with traint test split ratio. first index is train, second validate, third test.
        """
        self.ratio = ratio
        data = np.load(data_path)
        t = th.from_numpy(data["t"]).int()
        yf = th.from_numpy(data["yf"])
        y_cf = th.from_numpy(data["ycf"])
        self.size = t.size()[0]

        mask = t == 1
        ground_truth_cate = mask * (yf - y_cf) + ~mask * (y_cf - yf)

        super().__init__(
            th.from_numpy(np.transpose(data["x"], (0, 2, 1))).float(),
            th.from_numpy(data["t"]).int(),
            th.from_numpy(data["yf"]).float(),
            ground_truth_cate.float(),
            th.from_numpy(data["mu1"]).float(),
            th.from_numpy(data["mu0"]).float(),
        )

    def __iter__(
        self,
    ) -> Generator[
        Tuple[
            Tuple[th.Tensor, th.IntTensor, th.Tensor],
            Tuple[th.Tensor, th.IntTensor, th.Tensor],
            Tuple[th.Tensor, th.IntTensor, th.Tensor, th.Tensor],
            Tuple[th.Tensor, th.IntTensor, th.Tensor],
            Tuple[th.Tensor, th.IntTensor, th.Tensor],
        ],
        None,
        None,
    ]:
        for X, T, Y, ground_truth_cate, mu1, mu0 in zip(*self.tensors):
            (
                x_train,
                x_test,
                t_train,
                t_test,
                y_train,
                y_test,
                _,
                cate_test,
                _,
                mu1_test,
                _,
                mu0_test,
            ) = train_test_split(
                X.clone().detach(),
                T.clone().detach(),
                Y.clone().detach(),
                ground_truth_cate.clone().detach(),
                mu1.clone().detach(),
                mu0.clone().detach(),
                test_size=1 - self.ratio[0],
            )
            (
                x_val,
                x_test,
                t_val,
                t_test,
                y_val,
                y_test,
                _,
                cate_test,
                mu1_val,
                _,
                mu0_val,
                _,
            ) = train_test_split(
                x_test,
                t_test,
                y_test,
                cate_test,
                mu1_test,
                mu0_test,
                test_size=self.ratio[2] / (self.ratio[2] + self.ratio[1]),
            )
            yield (x_train, t_train, y_train), (x_val, y_val, t_val), (mu0_val, mu1_val), (
                x_test,
                y_test,
                t_test,
                cate_test,
            )

    def __len__(self):
        return self.size
