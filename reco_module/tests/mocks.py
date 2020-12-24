import torch
from torch.utils.data import Dataset, TensorDataset


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

