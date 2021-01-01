import itertools
import os
import pandas as pd
from collections import Counter
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.loggers import TensorBoardLogger
from .reco_dataset import Dataset


class BookingSequenceDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        """
        We use pandas to store the data

        Context data is stored as list of lists
        """
        df = pd.read_csv(os.path.join(self.data_dir, "booking_train_set.csv"))

        # Some indexes may be missing, we need to reindex
        self.nb_cities = len(set(df.city_id))
        self.index_to_cities = dict(enumerate(set(df.city_id)))
        self.cities_to_index = {v: k for k, v in self.index_to_cities.items()}
        country_per_city_tuples = dict(df.groupby(df.city_id).\
                                       agg({"hotel_country": "unique"}).\
                                       itertuples(index=True))
        self.city_to_country = dict(map(lambda p: (p[0], p[1][0]), country_per_city_tuples.items()))

        # Transform some cols before computing the features for the problem
        self.build_df_features(df)

        # Now we do a split of the training set based user_ids
        train_users, valid_users = self.build_train_val_split(df)
        self.train_set = list(self.build_trip_features(df[df.user_id.isin(train_users)]))
        self.valid_set = list(self.build_trip_features(df[df.user_id.isin(valid_users)]))

    def build_train_val_split(self, df, percent=0.9):
        users = set(df.user_id.values)
        nb_users = len(users)
        train_nb_users = int(percent * nb_users)
        print(f"Picking {train_nb_users} out of {nb_users}")
        train_users = np.random.choice(np.array(list(users)).flatten(), train_nb_users, replace=False)
        valid_users = users.difference(set(train_users))
        return train_users, valid_users

    def build_df_features(self, df):
        df["city_id"] = df["city_id"].apply(lambda x: self.cities_to_index[x])

    @staticmethod
    def build_trip_features(df):

        def get_non_label(grouped_x):
            data = list(grouped_x.values)
            if len(grouped_x.values) < 2:
                return [-1]
            else:
                return data[:-1]

        def get_booking_month(grouped_x):
            data = grouped_x.values
            if len(grouped_x.values) < 2:
                return None
            else:
                return pd.Timestamp(data[-2]).month

        def split_feat_label(grouped_x):
            data = list(grouped_x.values)
            return data[:-1], data[-1]

        per_trip = df.groupby(df.utrip_id). \
            agg({"city_id": split_feat_label, "checkout": get_booking_month, "affiliate_id": get_non_label})

        return per_trip.itertuples(index=True)

    def build_sequence_tensor(self, sequences):
        sequences = list(filter(lambda x: len(x.city_id[0]) > 0, sequences))
        city_sequences = map(lambda x: x.city_id, sequences)
        cities, label = zip(*list(city_sequences))
        dates = map(lambda x: x.checkout, sequences)
        affiliate_ids = map(lambda x: x.affiliate_id[-1], sequences)
        return cities, label, list(dates), list(affiliate_ids)

    def my_collate(self, batch):
        """
        Data format is city_seq, label, feature_1, feature_2
        :param batch:
        :return:
        """
        n_features = len(batch[0])

        data = [torch.LongTensor(item[0]) for item in batch]
        sizes = [len(item[0]) for item in batch]
        last_city = torch.LongTensor([item[0][-1] for item in batch])
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)

        additional_features = []
        for i in range(2, n_features):
            additional_features.append(torch.LongTensor([item[i] for item in batch]))

        return [(data, sizes, last_city, *additional_features), target]

    def train_dataloader(self):
        X, Y, *other_features = self.build_sequence_tensor(self.train_set)
        return DataLoader(Dataset(X, Y, *other_features), collate_fn=self.my_collate, batch_size=self.batch_size,
                          shuffle=True, num_workers=0)

    def val_dataloader(self):
        X, Y, *other_features = self.build_sequence_tensor(self.valid_set)
        return DataLoader(Dataset(X, Y, *other_features), collate_fn=self.my_collate, batch_size=self.batch_size,
                          shuffle=True)

    def test_dataloader(self):
        return self.val_dataloader()

