import os
import torch
from torch.utils.data import Dataset, TensorDataset
import pytorch_lightning as pl
from reco_module.mf.knn_learner import KnnLearner
from reco_module.mf.mf_learner import MatrixFactorization
from reco_module.dataset.sequential_dataset import BookingSequenceDataModule
from reco_module.dataset.reco_dataset import BookingTripRecoDataModule


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class DummyRecoDs(CustomTensorDataset):

    def __init__(self, n_classes=10, n_samples=1000):
        X_train = torch.randint(n_classes, (n_samples, 2))
        y_train = torch.randint(n_classes, (n_samples, 1))
        super().__init__(tensors=(X_train, y_train))


def create_dummy_mf_model(data_path, country_mode=False, max_lines=10000, embedding_size=50):

    reco_dataset = BookingTripRecoDataModule(data_path, 256, max_rows=max_lines, country_mode=country_mode)
    reco_dataset.setup()
    n_items = reco_dataset.nb_cities if not country_mode else reco_dataset.nb_countries

    mf_trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=20)
    mf = MatrixFactorization(n_items, 0.01, embedding_size)

    # Where to save the dummy model
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    sub_name = "my_city_model.chkpt" if not country_mode else "my_country_model.chkpt"
    save_path = os.path.join(THIS_DIR, sub_name)
    # Gen a country model
    mf_trainer.fit(mf, reco_dataset.train_dataloader())
    mf_trainer.save_checkpoint(save_path)

    return save_path
