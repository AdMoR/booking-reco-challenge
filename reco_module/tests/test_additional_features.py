from unittest import TestCase
import os

import pytorch_lightning as pl
from .mocks import create_dummy_mf_model
from reco_module.mf.knn_learner import KnnLearner
from reco_module.mf.mf_learner import MatrixFactorization
from reco_module.dataset.sequential_dataset import BookingSequenceDataModule
from reco_module.dataset.reco_dataset import BookingTripRecoDataModule


class TestDatasetRecoModelIntegration(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.embedding_size = 50
        cls.max_lines = 10000
        data_path = "/Users/a.morvan/Documents/code_dw/booking-reco-challenge/data"

        cls.city_save_path = create_dummy_mf_model(data_path, country_mode=False)
        cls.country_save_path = create_dummy_mf_model(data_path, country_mode=True,
                                                      embedding_size=int(cls.embedding_size / 2))

        cls.dataset = BookingSequenceDataModule(data_path, 1024, max_rows=cls.max_lines)
        cls.dataset.setup()
        cls.n_cities = cls.dataset.nb_cities
        cls.n_countries = cls.dataset.nb_countries

    def test_one_epoch(self):
        knn_learner = KnnLearner(self.n_cities, self.n_countries, self.city_save_path, self.country_save_path,
                                 self.embedding_size, 0.001,
                                 nb_affiliates=len(self.dataset.index_to_affiliates))
        trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=20)
        trainer.fit(knn_learner, self.dataset.train_dataloader(), self.dataset.val_dataloader())

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.city_save_path)
        os.remove(cls.country_save_path)
