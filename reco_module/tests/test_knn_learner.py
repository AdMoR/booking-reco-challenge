from unittest import TestCase
import os

import torch
from torch import nn
from reco_module.mf.knn_learner import KnnLearner


class TestKnnLearner(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_items = 38020
        cls.embedding_size = 10
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.test_data = os.path.join(THIS_DIR, "example.ckpt")

    def setUp(self):
        self.model = KnnLearner(self.n_items, embedding_size=self.embedding_size, weight_path=self.test_data)

    def test_forward(self):
        self.model.forward([[1, 2, 3], [1, 2]],
                           torch.LongTensor([1, 5, 9, 0]))

    def test_train_step(self):
        x = [[1, 2, 3]]
        y = [0]
        self.model.training_step((x, y), 0)

