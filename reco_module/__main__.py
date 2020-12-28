import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from .utils.embedding_callback import EmbeddingLoggerCallBack
from .mf.mf_learner import MatrixFactorization
from .mf.knn_learner import KnnLearner
from .dataset.reco_dataset import BookingTripRecoDataModule
from .dataset.sequential_dataset import BookingSequenceDataModule


def main(max_epochs=20, embedding_size=50, lr=1e-3, save_path="./my_new_mf_model.chkpt"):

    dataset = BookingTripRecoDataModule("/Users/a.morvan/Documents/code_dw/booking-reco-challenge/data", 256)
    dataset.setup()

    if not os.path.exists(save_path):

        dl = dataset.train_dataloader()
        logger = TensorBoardLogger("tb_logs", name="mf_model")
        new_trainer = pl.Trainer(max_epochs=max_epochs, progress_bar_refresh_rate=20,
                                 check_val_every_n_epoch=1, logger=logger,
                                 val_check_interval=0.25,
                                 callbacks=[EmbeddingLoggerCallBack(list(dataset.city_to_country.values()))])

        mf = MatrixFactorization(dataset.nb_cities, lr, embedding_size)

        new_trainer.fit(mf, dl)
        new_trainer.test(mf, dl)

        save_path = "./my_mf_model.chkpt"
        new_trainer.save_checkpoint(save_path)

    sequence_dataset = BookingSequenceDataModule("/Users/a.morvan/Documents/code_dw/booking-reco-challenge/data", 256)
    sequence_dataset.setup()
    knn_learner = KnnLearner(dataset.nb_cities, embedding_size, save_path, lr)

    logger = TensorBoardLogger("tb_logs", name="sequence_model")
    trainer = pl.Trainer(max_epochs=max_epochs, progress_bar_refresh_rate=20,
                         check_val_every_n_epoch=1, logger=logger,
                         val_check_interval=0.25,
                         callbacks=[EmbeddingLoggerCallBack(list(dataset.city_to_country.values()))])

    trainer.fit(knn_learner, sequence_dataset.train_dataloader())
    trainer.test(knn_learner, sequence_dataset.test_dataloader())


if __name__ == "__main__":
    main()
