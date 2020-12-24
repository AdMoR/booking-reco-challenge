from pytorch_lightning.callbacks import Callback


class EmbeddingLoggerCallBack(Callback):

    def __init__(self, country_list):
        self.country_list = country_list
        self.i = 0

    def on_epoch_end(self, trainer, pl_module):
        emb = list(pl_module.embeddings.parameters())[0]
        trainer.logger.experiment.add_embedding(emb, metadata=list(self.country_list), global_step=self.i)
        self.i += 1