import torch as th
import numpy as np
from torch.utils.data import TensorDataset

class ihdpDataset(TensorDataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        x_reshape = data['x'].reshape((data['x'].shape[0] * data['x'].shape[2], data['x'].shape[1]))
        t_reshape = data['t'].reshape((data['t'].shape[0] * data['t'].shape[1], 1))
        y_reshape = data['yf'].reshape((data['yf'].shape[0] * data['yf'].shape[1], 1))
        # self.X = torch.tensor(x_reshape, dtype=torch.float32)
        # self.T = torch.tensor(t_reshape, dtype=torch.float32)
        # self.Y = torch.tensor(y_reshape, dtype=torch.float32)
        super().__init__(th.from_numpy(x_reshape),
                         th.from_numpy(t_reshape).int(),
                         th.from_numpy(y_reshape))

