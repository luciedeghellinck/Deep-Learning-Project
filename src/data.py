import torch as th
import numpy as np
from torch.utils.data import TensorDataset

class ihdpDataset(TensorDataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        x_reshape = np.swapaxes(np.swapaxes(data['x'], 0, 2), 1, 2)
        t_reshape = np.swapaxes(data['t'], 0, 1)
        y_reshape = np.swapaxes(data['yf'], 0, 1)
        # x_reshape = data['x'].reshape((data['x'].shape[0] * data['x'].shape[2], data['x'].shape[1]))
        # t_reshape = data['t'].reshape((data['t'].shape[0] * data['t'].shape[1], 1))
        # y_reshape = data['yf'].reshape((data['yf'].shape[0] * data['yf'].shape[1], 1))
        self.X = th.tensor(x_reshape, dtype=th.float32)  # realisation * input per realisation * feature
        self.T = th.tensor(t_reshape, dtype=th.int32)  # realisation * input per realisation
        self.Y = th.tensor(y_reshape, dtype=th.float32)  # realisation * input per realisation
        # super().__init__(th.from_numpy(x_reshape),
        #                  th.from_numpy(t_reshape).int().squeeze(-1),
        #                  th.from_numpy(y_reshape).squeeze(-1))
        super().__init__()

