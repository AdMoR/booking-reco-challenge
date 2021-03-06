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

    def __init__(self, n_cities, n_countries, city_weight_path, country_weight_path, embedding_size=50, lr=1e-4, layer_size=(20, 10),
                 dummy_model=None, multiplier=2, n_affiliates=3065, emb_aff_size=10, n_dates=13, emb_date_size=2,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        country_emb_size = int(embedding_size / 2)
        print("----> Params : ", n_cities, n_countries, city_weight_path, country_weight_path)
        self.n_affiliates = n_affiliates
        self.lr = lr
        self.city_embeddings_model = MatrixFactorization.\
            load_from_checkpoint(checkpoint_path=city_weight_path, n_items=n_cities,
                                 lr=self.lr, embedding_size=embedding_size).embeddings
        self.country_embeddings_model = MatrixFactorization.\
            load_from_checkpoint(checkpoint_path=country_weight_path, n_items=n_countries,
                                 lr=self.lr, embedding_size=country_emb_size).embeddings
        self.n_cities = n_cities
        self.n_countries = n_countries

        entry_size = embedding_size * multiplier + 0 * emb_aff_size + emb_date_size + country_emb_size
        self.user_tower = nn.Sequential(*[nn.Linear(entry_size, layer_size[0]),
                                          nn.ReLU(),
                                          nn.Linear(layer_size[0], layer_size[1])])
        self.item_tower = nn.Sequential(*[nn.Linear(embedding_size, layer_size[0]),
                                          nn.ReLU(),
                                          nn.Linear(layer_size[0], layer_size[1])])
        self.dummy_model = dummy_model
        self.affiliate_embedding = nn.Embedding(n_affiliates, emb_aff_size)
        self.date_embedding = nn.Embedding(n_dates, emb_date_size)

    def forward(self, xs_user, sizes, last_city=None, date_month=None, affiliate_id=None, countries=None):
        """
        :param xs_user: list of list -> each list represent the item_id of a user
        :param sizes: list(int) size of each list before padding
        :param last_city: Tensor([city_ids]) contains the last city of the user for each current trip
        :param date_month: Tensor([month_id])
        :param affiliate_id: Tensor([aff_ids])
        :param countries: tensor(country_id)
        :return:
        """
        # During the training, we will look for the dot product btw user vector and all item vectors
        item_indexes = torch.LongTensor(np.arange(self.n_cities))
        item_embeddings = self.city_embeddings_model(item_indexes)
        final_item_embeddings = self.item_tower(item_embeddings)

        seq_sizes = torch.FloatTensor(sizes)
        X = pad_sequence(xs_user, batch_first=False, padding_value=self.n_cities)
        user_embeddings = self.city_embeddings_model(X)
        user_batch = torch.sum(user_embeddings, dim=0) / seq_sizes.reshape(-1, 1)

        if last_city is not None:
            last_city_embeddings = self.city_embeddings_model(last_city)
            user_batch = torch.cat([user_batch, last_city_embeddings], dim=1)
        if affiliate_id is not None:
            print("--->", torch.max(affiliate_id), self.n_affiliates)
            affiliate_id_embedding = self.affiliate_embedding(affiliate_id)
            user_batch = torch.cat([user_batch, affiliate_id_embedding], dim=1)
        if date_month is not None:
            date_embedding = self.date_embedding(date_month)
            user_batch = torch.cat([user_batch, date_embedding], dim=1)
        if countries is not None:
            country_emb = self.country_embeddings_model(countries)
            user_batch = torch.cat([user_batch, country_emb], dim=1)

        print("userbatch size : ")
        user_features = self.user_tower(user_batch)
        scores = user_features @ final_item_embeddings.transpose(1, 0)

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
        (x, sizes, last_city, *other_features), y = batch

        try:
            date_month, affiliate_batch, countries = other_features
        except Exception:
            print("Could not parse other features")
            date_month = None
            affiliate_batch = None
            countries = None

        scores = self.forward(x, sizes, last_city, date_month, None, countries)
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

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        parameters = itertools.chain(
            self.city_embeddings_model.parameters(),
            self.user_tower.parameters(),
            self.item_tower.parameters(),
            self.affiliate_embedding.parameters(),
            self.date_embedding.parameters()
        )
        optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=1e-5)
        return optimizer
