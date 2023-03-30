import torch as th
from torch.utils.data import DataLoader, TensorDataset

from src.data import ihdpDataset


class TestDataset:
    def test_loading_ihdp(self):
        dataset = ihdpDataset("../dataset/ihdp_npci_1-100.test.npz")
        assert (
            dataset.X.size()[0] == 100
            and dataset.X.size()[1] == 75
            and dataset.X.size()[2] == 25
        )
        assert dataset.T.size()[0] == 100 and dataset.T.size()[1] == 75
        assert dataset.Y.size()[0] == 100 and dataset.Y.size()[1] == 75
        subDataset = (
            th.Tensor(dataset.X[0]),
            th.tensor(dataset.T[0]),
            th.Tensor(dataset.Y[0]),
        )

        loader = DataLoader(TensorDataset(*subDataset), batch_size=64)

        for item in loader:
            assert isinstance(item, list)
            assert len(item) == 3
            x, t, y = item

            assert isinstance(x, th.Tensor)
            assert isinstance(t, th.IntTensor)
            assert isinstance(y, th.Tensor)
            assert len(x.size()) == 2
            assert len(t.size()) == 1
            assert len(y.size()) == 1
