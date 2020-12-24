from unittest import TestCase
from torch.utils.data import DataLoader

from reco_module.dataset.reco_dataset import BookingTripRecoDataModule, Dataset
from reco_module.dataset.sequential_dataset import BookingSequenceDataModule


class TestDataset(TestCase):

    def setUp(self):
        self.dataset = BookingTripRecoDataModule("/Users/a.morvan/Documents/code_dw/booking-reco-challenge/data", 256)

    def test_reco_dataset_setup(self):
        self.dataset.setup()

    def test_reco_dataset_examples(self):
        self.dataset.setup()
        X, Y = self.dataset.build_neg_pairs(self.dataset.train_set_pairs[:100], 100)
        dl = DataLoader(Dataset(X, Y), batch_size=self.dataset.batch_size, shuffle=True)


class TestSequenceDataset(TestCase):

    def setUp(self):
        self.dataset = BookingSequenceDataModule("/Users/a.morvan/Documents/code_dw/booking-reco-challenge/data", 256)

    def test_reco_dataset_setup(self):
        self.dataset.setup()

    def test_reco_dataset_examples(self):
        self.dataset.setup()
        ds = self.dataset.train_dataloader()
