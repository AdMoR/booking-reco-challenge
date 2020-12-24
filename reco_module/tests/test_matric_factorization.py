from unittest import TestCase
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from reco_module.mf.mf_learner import MatrixFactorization
from .mocks import DummyRecoDs


class TestMatrixFactorization(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf_alg = MatrixFactorization(10, 5)

    def test_mf_forward(self):
        self.mf_alg.forward(
            torch.LongTensor(np.array([[1, 2], [1, 3]]))
        )

    def test_mf_backward(self):
        batch = (torch.LongTensor([[9, 8], [1, 8], [3, 4]]),
                 torch.LongTensor([2, 3, 0]))
        self.mf_alg.training_step(batch, 0)

    def test_training(self):
        ds = DummyRecoDs(100, 22000)
        dl = DataLoader(ds, batch_size=32)

        # init model
        mf = MatrixFactorization(100)

        # Initialize a trainer
        trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=20)

        # Train the model ⚡
        trainer.fit(mf, dl)
