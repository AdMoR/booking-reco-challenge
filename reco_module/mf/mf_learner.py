import torch
from torch import nn
import pytorch_lightning as pl
from argparse import ArgumentParser


class MatrixFactorization(pl.LightningModule):

    def __init__(self, n_items, lr=1e-3, embedding_size=50, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.n_items = n_items
        # We add an additional dim to allow to query efficiently variable size user history
        self.embeddings = nn.Embedding(self.n_items + 1, embedding_size, padding_idx=-1)
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, x):
        u = self.embeddings(x[:, 0])
        v = self.embeddings(x[:, 1])
        x_hat = self.sim(u, v)
        return x_hat
    
    def weighted_cross_entropy(self, x, y, w, eps=1e-8):
        elmt_wise_loss = (y * torch.log(x + eps) + (1 - y) * torch.log(1 - x + eps))
        return -(w * elmt_wise_loss).sum() / w.sum()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = 0.5 + 0.5 * self.forward(x)
        y_prime = torch.where(y > 0, torch.ones_like(y), torch.zeros_like(y))
        weights = torch.where(y > 0, y, torch.ones_like(y)).to(torch.float32)
        loss = self.weighted_cross_entropy(x_hat, y_prime, weights)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('val_loss', loss)
        # --------------------------

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.embeddings.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

