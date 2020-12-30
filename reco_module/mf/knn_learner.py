import os
from argparse import ArgumentParser
import itertools
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import pytorch_lightning as pl

from reco_module.mf.mf_learner import MatrixFactorization
from reco_module.utils.dummy_reco import MaxCoocModel


class KnnLearner(pl.LightningModule):

    def __init__(self, n_items, city_weight_path, embedding_size=50, lr=1e-4, layer_size=(20, 10),
                 dummy_model=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print("Params : ", n_items, embedding_size, city_weight_path)
        self.lr = lr
        self.embeddings_model = MatrixFactorization.\
            load_from_checkpoint(checkpoint_path=city_weight_path, n_items=n_items,
                                 lr=self.lr, embedding_size=embedding_size)
        self.n_items = n_items

        self.user_tower = nn.Sequential(*[nn.Linear(embedding_size, layer_size[0]),
                                          nn.ReLU(),
                                          nn.Linear(layer_size[0], layer_size[1])])
        self.item_tower = nn.Sequential(*[nn.Linear(embedding_size, layer_size[0]),
                                          nn.ReLU(),
                                          nn.Linear(layer_size[0], layer_size[1])])

        self.dummy_model = dummy_model

    def forward(self, xs_user, sizes):
        """

        :param xs_user: list of list -> each list represent the item_id of a user
        :param sizes: list(int) size of each list before padding
        :return:
        """
        # During the training, we will look for the dot product btw user vector and all item vectors
        item_indexes = torch.LongTensor(np.arange(self.embeddings_model.n_items))
        item_embeddings = self.embeddings_model.embeddings(item_indexes)
        self.final_item_embeddings = self.item_tower(item_embeddings)

        seq_sizes = torch.FloatTensor(sizes)
        X = pad_sequence(xs_user, batch_first=False, padding_value=self.n_items)
        user_embeddings = self.embeddings_model.embeddings(X)
        user_features = self.user_tower(user_embeddings)

        user_batch = torch.sum(user_features, dim=0) / seq_sizes.reshape(-1, 1)
        scores = user_batch @ self.final_item_embeddings.transpose(1, 0)

        return scores

    def weighted_cross_entropy(self, x, y, w=None, eps=1e-8):
        elmt_wise_loss = torch.log(
            x[torch.LongTensor(np.arange(y.shape[0])), y] + eps
        )
        if w is None:
            return -elmt_wise_loss.mean()
        else:
            return -(w * elmt_wise_loss).sum() / w.sum()

    def training_step(self, batch, batch_idx):
        (x, sizes), y = batch
        scores = self.forward(x, sizes)
        loss = self.weighted_cross_entropy(F.softmax(scores, dim=1), torch.LongTensor(y))
        self.log('loss', loss)
        topk_matches = torch.sum(torch.topk(scores, k=4, dim=1).indices == y.reshape(-1, 1))
        self.log('top_k matches', topk_matches / y.shape[0])

        if self.dummy_model:
            my_batch = [v[-1] for v in x]
            dummy_scores = torch.Tensor(self.dummy_model.batch_scores(my_batch))
            topk_matches = torch.sum(torch.topk(dummy_scores, k=4, dim=1).indices == y.reshape(-1, 1))
            self.log('top_k matches for dummy', topk_matches / y.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss', loss)
        # --------------------------

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        parameters = itertools.chain(
            self.embeddings_model.embeddings.parameters(),
            self.user_tower.parameters(),
            self.item_tower.parameters(),
        )
        optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=1e-5)
        return optimizer
