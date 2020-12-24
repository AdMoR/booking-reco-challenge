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

        1 - We extract te positive pair of co-occurance of cities among user trips
        2 - Based on the pos pairs, we generate negative example with a neg to pos ratio = 5
        3 - We format this to Nx2 and Nx1 vectors
        """
        df = pd.read_csv(os.path.join(self.data_dir, "booking_train_set.csv"), nrows=999999)

        # Some indexes may be missing, we need to reindex
        self.nb_cities = len(set(df.city_id))
        self.index_to_cities = dict(enumerate(set(df.city_id)))
        self.cities_to_index = {v: k for k, v in self.index_to_cities.items()}

        # Additional infos
        country_per_city_tuples = dict(df.groupby(df.city_id).agg({"hotel_country": "unique"}). \
                                       itertuples(index=True))
        self.city_to_country = dict(map(lambda p: (p[0], p[1][0]), country_per_city_tuples.items()))

        # Now we do a split of the training set based user_ids
        users = set(df.user_id.values)
        nb_users = len(users)
        train_nb_users = int(0.9 * nb_users)
        print(f"Picking {train_nb_users} out of {nb_users}")
        train_users = np.random.choice(np.array(list(users)).flatten(), train_nb_users, replace=False)
        valid_users = users.difference(set(train_users))

        def from_df_to_sequence(sub_df):
            per_trip = sub_df.groupby(df.utrip_id). \
                agg({"city_id": list})
            city_lists = per_trip["city_id"].apply(lambda x: (x[:-1], x[-1]))

            return list(city_lists.itertuples(index=True))

        self.train_set = list(from_df_to_sequence(df[df.user_id.isin(train_users)]))
        self.valid_set = list(from_df_to_sequence(df[df.user_id.isin(valid_users)]))

    def build_sequence_tensor(self, sequences):
        xs, ys = zip(*sequences)
        X = pad_sequence(list(map(torch.LongTensor, xs)), padding_value=-1)
        return X, torch.LongTensor(ys)

    def train_dataloader(self):
        X, Y = self.build_sequence_tensor(self.train_set)
        return DataLoader(Dataset(X, Y), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        X, Y = self.build_sequence_tensor(self.valid_set)
        return DataLoader(Dataset(X, Y), batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return self.val_dataloader()

