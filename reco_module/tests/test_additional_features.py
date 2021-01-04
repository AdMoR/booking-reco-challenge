from unittest import TestCase
import os

import torch
from torch import nn
import pytorch_lightning as pl
from reco_module.mf.knn_learner import KnnLearner
from reco_module.mf.mf_learner import MatrixFactorization
from reco_module.dataset.sequential_dataset import BookingSequenceDataModule
from reco_module.dataset.reco_dataset import BookingTripRecoDataModule


class TestDatasetRecoModelIntegration(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedding_size = 50
        cls.max_lines = 10000


        data_path = "/home/amor/Documents/code_dw/booking_challenge/data"
        cls.reco_dataset = BookingTripRecoDataModule(data_path, 256, max_rows=cls.max_lines)
        cls.reco_dataset.setup()
        cls.n_items = cls.reco_dataset.nb_cities

        mf_trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=20)
        mf = MatrixFactorization(cls.reco_dataset.nb_cities, 0.01, cls.embedding_size)

        # Where to save the dummy model
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.save_path = os.path.join(THIS_DIR, "my_model.chkpt")

        # Gen a model
        mf_trainer.fit(mf, cls.reco_dataset.train_dataloader())
        mf_trainer.save_checkpoint(cls.save_path)

        cls.dataset = BookingSequenceDataModule(data_path, 1024, max_rows=cls.max_lines)


    def test_one_epoch(self):
        self.dataset.setup()

        knn_learner = KnnLearner(self.n_items, self.save_path, self.embedding_size, 0.001, 
                                 nb_affiliates=len(self.dataset.index_to_affiliates))
        trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=20)
        trainer.fit(knn_learner, self.dataset.train_dataloader())