from unittest import TestCase
import os

import torch
from torch import nn
from reco_module.mf.knn_learner import KnnLearner
from .mocks import create_dummy_mf_model
from reco_module.dataset.sequential_dataset import BookingSequenceDataModule


class TestKnnLearner(TestCase):

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

    def setUp(self):
        self.model = KnnLearner(self.n_cities, self.n_countries,
                                embedding_size=self.embedding_size,
                                city_weight_path=self.city_save_path,
                                country_weight_path=self.country_save_path,
                                multiplier=1)

    def test_forward(self):
        x = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2])]
        sizes = torch.FloatTensor([3, 2])
        dates = torch.LongTensor([1, 9])
        countries = torch.LongTensor([1, 2])
        self.model.forward(x, sizes, None, dates, None, countries)

    def test_forward_with_last_city(self):
        self.model = KnnLearner(self.n_cities, self.n_countries,
                                embedding_size=self.embedding_size,
                                city_weight_path=self.city_save_path,
                                country_weight_path=self.country_save_path,
                                multiplier=2)
        x = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2])]
        last_city = torch.LongTensor([3, 2])
        sizes = torch.FloatTensor([3, 2])
        dates = torch.LongTensor([1, 9])
        countries = torch.LongTensor([1, 2])
        self.model.forward(x, sizes, last_city, dates, None, countries)

    def test_train_step(self):
        x = [torch.LongTensor([1, 2, 3])]
        y = torch.LongTensor([0])
        sizes = torch.FloatTensor([3])
        dates = torch.LongTensor([1])
        countries = torch.LongTensor([1])
        self.model.training_step(((x, sizes, None, dates, None, countries), y), 0)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.city_save_path)
        os.remove(cls.country_save_path)
