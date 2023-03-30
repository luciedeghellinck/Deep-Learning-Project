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


class ihdpDataset_CF(TensorDataset):
    #load counter factual outcome, protential outcome to calculate ground truth tau, and cfr
    def __init__(self, data_path):
        data = np.load(data_path)
        x_reshape = np.swapaxes(np.swapaxes(data['x'], 0, 2), 1, 2)
        t_reshape = np.swapaxes(data['t'], 0, 1)
        y_f_reshape = np.swapaxes(data['yf'], 0, 1)
        y_cf_reshape = np.swapaxes(data['ycf'], 0, 1)
        mu1_reshape = np.swapaxes(data['mu1'], 0, 1)
        mu0_reshape = np.swapaxes(data['mu0'], 0, 1)

        self.X = th.tensor(x_reshape, dtype=th.float32)
        self.T = th.tensor(t_reshape, dtype=th.float32)
        self.Y_f = th.tensor(y_f_reshape, dtype=th.float32)
        self.Y_cf = th.tensor(y_cf_reshape, dtype=th.float32)
        self.mu1 = th.tensor(mu1_reshape, dtype=th.float32)
        self.mu0 = th.tensor(mu0_reshape, dtype=th.float32)
        super().__init__()
