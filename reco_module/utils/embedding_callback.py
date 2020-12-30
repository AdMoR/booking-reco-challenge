from pytorch_lightning.callbacks import Callback
import torch
import numpy as np

class EmbeddingLoggerCallBack(Callback):

    def __init__(self, country_list):
        self.country_list = country_list
        self.i = 0

    def on_epoch_end(self, trainer, pl_module):
        try:
            # During the training, we will look for the dot product btw user vector and all item vectors
            item_indexes = torch.LongTensor(np.arange(pl_module.embeddings_model.n_items + 1))
            item_embeddings = pl_module.embeddings_model.embeddings(item_indexes)
            emb = pl_module.item_tower(item_embeddings)
        except Exception:
            emb = list(pl_module.embeddings.parameters())[0]
        print(emb.shape, len(self.country_list))
        trainer.logger.experiment.add_embedding(emb, metadata=list(self.country_list) + ["Dummy"], global_step=self.i)
        self.i += 1
