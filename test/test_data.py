import torch as th
from torch.utils.data import DataLoader

from src.data import Dataset

class TestDataset:

    def test_loading_ihdp(self):
        dataset = Dataset("../ihdp_npci_1-100.test.npz")
        loader = DataLoader(dataset, batch_size=64)

        for item in loader:
            assert isinstance(item, list)
            assert len(item) == 3
            x, t, y = item

            assert isinstance(x, th.Tensor)
            assert isinstance(t, th.IntTensor)
            assert isinstance(y, th.Tensor)
            print(x.size())
            assert len(x.size()) == 2
            assert len(t.size()) == 2
            assert len(y.size()) == 2