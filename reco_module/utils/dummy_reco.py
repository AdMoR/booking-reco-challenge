import numpy as np
from scipy.sparse import dok_matrix
import itertools
import pandas as pd


class MaxCoocModel:

    def __init__(self, data_path):

        df = pd.read_csv(data_path, nrows=10000)
        self.index_to_cities = dict(enumerate(set(df.city_id)))
        self.cities_to_index = {v: k for k, v in self.index_to_cities.items()}

        # Next we emit positive coocuring pairs
        def from_user_list_to_cooc_pairs(sub_df):
            per_trip = sub_df.groupby(sub_df.utrip_id). \
                agg({"city_id": "unique", "user_id": "count", "hotel_country": "unique"})

            per_trip["nb_countries"] = per_trip["hotel_country"].apply(len)
            per_trip["nb_cities"] = per_trip["city_id"].apply(len)

            def emit_pairs(cities):
                sorted_cities = sorted(map(lambda x: self.cities_to_index[x], cities))
                for i in range(len(cities)):
                    for j in range(i + 1, len(cities)):
                        yield (sorted_cities[i], sorted_cities[j])

            co_cities = per_trip["city_id"]
            return list(itertools.chain.from_iterable(map(emit_pairs, co_cities)))

        n_items = max(self.index_to_cities.keys()) + 1
        self.cooc_mat = dok_matrix((n_items, n_items), dtype=np.float32)

        coocs = from_user_list_to_cooc_pairs(df)
        for k in coocs:
            (i, j) = k
            self.cooc_mat[i, j] += 1
            self.cooc_mat[j, i] += 1

    def __call__(self, i):
        return self.predict(i)

    def predict(self, i, top=4):
        assert i in self.cities_to_index
        index = self.cities_to_index[i]
        results = np.argsort(self.cooc_mat[index, :].toarray())[0, -top:].tolist()
        return list(map(lambda x: self.index_to_cities[x], results))

    def scores(self, i):
        line = self.cooc_mat[i, :].toarray()
        return line / np.sum(line)

    def batch_scores(self, b):
        lines = self.cooc_mat[b, :].toarray()
        return lines / np.sum(lines, axis=1).reshape(-1, 1)
