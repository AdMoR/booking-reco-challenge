import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from .utils.embedding_callback import EmbeddingLoggerCallBack
from .mf.mf_learner import MatrixFactorization
from .dataset.reco_dataset import BookingTripRecoDataModule


def main(max_epochs=20, embedding_size=50, lr=1e-3):

    dataset = BookingTripRecoDataModule("/Users/a.morvan/Documents/code_dw/booking-reco-challenge/data", 256)
    dataset.setup()
    dl = dataset.train_dataloader()
    logger = TensorBoardLogger("tb_logs", name="my_model")
    new_trainer = pl.Trainer(max_epochs=max_epochs, progress_bar_refresh_rate=20,
                             check_val_every_n_epoch=1, logger=logger,
                             val_check_interval=0.25,
                             callbacks=[EmbeddingLoggerCallBack(list(dataset.city_to_country.values()))])

    mf = MatrixFactorization(dataset.nb_cities, lr, embedding_size)

    new_trainer.fit(mf, dl)
    new_trainer.test(mf, dl)


if __name__ == "__main__":
    main()
