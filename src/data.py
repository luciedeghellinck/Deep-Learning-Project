import torch as th
import pandas as pd
from torch.utils.data import TensorDataset


class Dataset(TensorDataset):
    """
    Model selection procedure for causal inference models

    Args:
     dataset: input file containing the features, treatments and outcomes
     (add file type)
    """

    def __init__(self, csv_path):
        ##Should I use a super init? I don't understand it's use in the assignements...
        """
           Takes a dataset as input and returns a parsed Torch tensor where each
           row {X_i, T_i, Y_i} represents the feature vector, the treatment type
           and the outcome for a particular person.
        """

        data = pd.read_csv(csv_path)

        # Each n row represents a person; the first column is the m dimensional float
        # feature vector, the second column is the 0/1 treatment type, and the
        # third column is the float outcome.
        super().__init__(th.tensor(data.iloc[:, 2:].values, dtype=th.float32),
                         th.tensor(data.iloc[:, 0].values, dtype=th.int64),
                         th.tensor(data.iloc[:, 1].values, dtype=th.float32))
