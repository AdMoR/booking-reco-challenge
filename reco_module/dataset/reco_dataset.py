import itertools
import os
import pandas as pd
from collections import Counter
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger


class Dataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.tensors = args
        length = len(self.tensors[0])
        if not all([len(t) == length for t in self.tensors]):
            raise Exception(f"Tensor do not have all the same size : {[len(t) for t in self.tensors]}")

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, i):
        return [t[i] for t in self.tensors]


class BookingTripRecoDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, country_mode: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.country_mode = country_mode

    def setup(self, stage=None):
        """
        We use pandas to store the data

        1 - We extract te positive pair of co-occurance of cities among user trips
        2 - Based on the pos pairs, we generate negative example with a neg to pos ratio = 5
        3 - We format this to Nx2 and Nx1 vectors
        """
        df = pd.read_csv(os.path.join(self.data_dir, "booking_train_set.csv"))

        # Some indexes may be missing, we need to reindex
        self.nb_cities = len(set(df.city_id))
        self.nb_countries = len(set(df.hotel_country))
        self.index_to_cities = dict(enumerate(set(df.city_id)))
        self.cities_to_index = {v: k for k, v in self.index_to_cities.items()}
        self.index_to_country = dict(enumerate(set(df.hotel_country)))
        self.country_to_index = {v: k for k, v in self.index_to_country.items()}
        country_per_city_tuples = dict(df.groupby(df.city_id).agg({"hotel_country": "unique"}). \
                                       itertuples(index=True))
        self.city_to_country = dict(map(lambda p: (p[0], p[1][0]), country_per_city_tuples.items()))

        # Now we do a split of the training set based user_ids
        train_users, valid_users = self.build_train_val_split(df)

        self.train_df = df[df.user_id.isin(train_users)]
        self.valid_df = df[df.user_id.isin(valid_users)]

    def build_cooc_func(self):
        def from_user_list_to_cooc_pairs(sub_df):
            per_trip = sub_df.groupby(sub_df.utrip_id). \
                agg({"city_id": "unique", "user_id": "count", "hotel_country": "unique"})

            def emit_pairs(indexes):
                conversion_dict = self.cities_to_index if not self.country_mode else self.country_to_index
                sorted_cities = sorted(map(lambda x: conversion_dict[x], indexes))
                for i in range(len(indexes)):
                    for j in range(i + 1, len(indexes)):
                        yield (sorted_cities[i], sorted_cities[j])

            if not self.country_mode:
                coocs = per_trip["city_id"]
            else:
                coocs = per_trip["hotel_country"]
            return list(itertools.chain.from_iterable(map(emit_pairs, coocs)))
        return from_user_list_to_cooc_pairs

    @property
    def train_set_pairs(self):
        from_user_list_to_cooc_pairs = self.build_cooc_func()
        return list(from_user_list_to_cooc_pairs(self.train_df))

    @property
    def valid_set_pairs(self):
        from_user_list_to_cooc_pairs = self.build_cooc_func()
        return list(from_user_list_to_cooc_pairs(self.valid_df))

    def build_train_val_split(self, df, percent=0.9):
        users = set(df.user_id.values)
        nb_users = len(users)
        train_nb_users = int(percent * nb_users)
        print(f"Picking {train_nb_users} out of {nb_users}")
        train_users = np.random.choice(np.array(list(users)).flatten(), train_nb_users, replace=False)
        valid_users = users.difference(set(train_users))
        return train_users, valid_users

    def build_neg_pairs(self, pos_pairs, n_classes, neg_rate=5):

        pair_counter = Counter(pos_pairs)
        pos_pairs_set = set(pos_pairs)
        total_pos_pairs = len(pos_pairs)

        print("Total pos pairs : ", total_pos_pairs)

        def sample_neg_pair():
            i = None
            j = None
            while i is None or j is None or (i, j) in pos_pairs_set or (j, i) in pos_pairs_set:
                i = np.random.randint(n_classes)
                j = np.random.randint(n_classes)
            return (i, j)

        neg_pairs = [sample_neg_pair() for _ in range(neg_rate * total_pos_pairs)]

        xs, ys = zip(*pair_counter.items())

        X_pos = np.array(xs).reshape((-1, 2))
        Y_pos = np.array(ys)

        X_neg = np.array(neg_pairs).reshape((-1, 2))
        Y_neg = np.zeros((X_neg.shape[0]))

        X = np.concatenate([X_pos, X_neg], axis=0)
        Y = np.concatenate([Y_pos, Y_neg])
        print(X.shape, Y.shape)
        return X, Y

    def train_dataloader(self):
        n_classes = self.nb_cities if not self.country_mode else self.nb_countries
        X, Y = self.build_neg_pairs(self.train_set_pairs, n_classes)
        return DataLoader(Dataset(X, Y), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        n_classes = self.nb_cities if not self.country_mode else self.nb_countries
        X, Y = self.build_neg_pairs(self.valid_set_pairs, n_classes)
        return DataLoader(Dataset(X, Y), batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return self.val_dataloader()

