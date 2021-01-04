import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from .utils.embedding_callback import EmbeddingLoggerCallBack
from .utils.dummy_reco import MaxCoocModel
from .mf.mf_learner import MatrixFactorization
from .mf.knn_learner import KnnLearner
from .dataset.reco_dataset import BookingTripRecoDataModule
from .dataset.sequential_dataset import BookingSequenceDataModule


def main(max_epochs=30, embedding_size=50, lr=5e-4, city_save_path="./my_city_mf_model.chkpt",
         country_save_path="./my_country_mf_model"):

    data_path = "/home/amor/Documents/code_dw/booking_challenge/data"
    dataset = BookingTripRecoDataModule(data_path, 256)
    dataset.setup()

    if not os.path.exists(city_save_path):

        dl = dataset.train_dataloader()
        logger = TensorBoardLogger("tb_logs", name="mf_city_model")
        new_trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, progress_bar_refresh_rate=20,
                                 check_val_every_n_epoch=1, logger=logger,
                                 val_check_interval=0.25,
                                 callbacks=[EmbeddingLoggerCallBack(list(dataset.city_to_country.values()))])

        mf = MatrixFactorization(dataset.nb_cities, lr, embedding_size)

        new_trainer.fit(mf, dataset.train_dataloader(), dataset.val_dataloader())
        new_trainer.test(mf, dataset.test_dataloader())
        new_trainer.save_checkpoint(city_save_path)

    if not os.path.exists(country_save_path):
        dataset.country_mode = True
        dl = dataset.train_dataloader()
        logger = TensorBoardLogger("tb_logs", name="mf_country_model")
        new_trainer = pl.Trainer(gpus=1, max_epochs=max_epochs, progress_bar_refresh_rate=20,
                                 check_val_every_n_epoch=1, logger=logger,
                                 val_check_interval=0.25,
                                 callbacks=[EmbeddingLoggerCallBack(list(dataset.index_to_country.values()))])

        mf = MatrixFactorization(dataset.nb_countries, lr, int(embedding_size / 2))

        new_trainer.fit(mf, dataset.train_dataloader(), dataset.val_dataloader())
        new_trainer.test(mf, dataset.test_dataloader())
        new_trainer.save_checkpoint(country_save_path)


    dummy_model = MaxCoocModel(data_path + "/booking_train_set.csv")
    sequence_dataset = BookingSequenceDataModule(data_path, 1024)
    sequence_dataset.setup()
    knn_learner = KnnLearner(dataset.nb_cities, city_save_path, embedding_size, lr, 
                             nb_affiliates=len(sequence_dataset.index_to_affiliates), 
                             dummy_model=dummy_model)

    logger = TensorBoardLogger("tb_logs", name="sequence_model")
    trainer = pl.Trainer(max_epochs=2 * max_epochs, progress_bar_refresh_rate=20,
                         check_val_every_n_epoch=1, logger=logger,
                         val_check_interval=0.25,
                         callbacks=[EmbeddingLoggerCallBack(list(dataset.city_to_country.values()))])

    trainer.fit(knn_learner, sequence_dataset.train_dataloader(), sequence_dataset.val_dataloader())
    trainer.test(knn_learner, sequence_dataset.test_dataloader())


if __name__ == "__main__":
    main()
