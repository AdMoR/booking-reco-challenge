from unittest import TestCase
import os

import torch
from torch import nn
from reco_module.mf.knn_learner import KnnLearner


class TestKnnLearner(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_items = 38020
        cls.embedding_size = 50
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.test_data = os.path.join(THIS_DIR, "model.chkpt")

    def setUp(self):
        self.model = KnnLearner(self.n_items, embedding_size=self.embedding_size, city_weight_path=self.test_data,
                                multiplier=1)

    def test_forward(self):
        x = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2])]
        sizes = torch.FloatTensor([3, 2])
        dates = torch.LongTensor([1, 9])
        affiliate_ids = torch.LongTensor([1, 99])
        self.model.forward(x, sizes, None, dates, affiliate_ids)

    def test_forward_with_last_city(self):
        self.model = KnnLearner(self.n_items, embedding_size=self.embedding_size, city_weight_path=self.test_data,
                                multiplier=2)
        x = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2])]
        last_city = torch.LongTensor([3, 2])
        sizes = torch.FloatTensor([3, 2])
        dates = torch.LongTensor([1, 9])
        affiliate_ids = torch.LongTensor([1, 99])
        self.model.forward(x, sizes, last_city, dates, affiliate_ids)

    def test_train_step(self):
        self.model = KnnLearner(self.n_items, embedding_size=self.embedding_size, city_weight_path=self.test_data,
                                multiplier=1, n_affiliates=100)
        x = [torch.LongTensor([1, 2, 3])]
        y = torch.LongTensor([0])
        sizes = torch.FloatTensor([3])
        dates = torch.LongTensor([1])
        affiliate_ids = torch.LongTensor([99])
        self.model.training_step(((x, sizes, None, dates, affiliate_ids), y), 0)

