from unittest import TestCase

import torch

from reco_module.utils.dummy_reco import MaxCoocModel


class TestDummyModel(TestCase):

    def setUp(self):
        self.dummy_model = MaxCoocModel("/Users/a.morvan/Documents/code_dw/booking-reco-challenge/data/booking_train_set.csv")

    def test_dummy_pred(self):
        x = [torch.LongTensor([5, 8]), torch.LongTensor([5, 8, 33])]
        y = torch.LongTensor([1, 2])

        batch = [v[-1] for v in x]
        dummy_scores = torch.Tensor(self.dummy_model.batch_scores(batch))
        topk_matches = torch.sum(torch.topk(dummy_scores, k=4, dim=1).indices == y.reshape(-1, 1))
        print('top_k matches for dummy', topk_matches / y.shape[0])
