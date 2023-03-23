import pytest
import torch as th
from src.loss import IPM


class TestLoss:

    def test_hypothesis(self):
        ipm = IPM