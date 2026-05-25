import lightning as L
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics.classification import (BinaryAccuracy, BinaryAUROC,
                                         BinaryF1Score, BinaryPrecision,
                                         BinaryRecall)

from torch_models.src.model.mlp import MLPClassifierBinary


class LightningMLPClassifierBinary(L.LightningModule):
    def __init__(self, input_dim, lr, weight_decay) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = MLPClassifierBinary(input_dim)
        self.loss_fn = nn.BCELoss()
        self.lr = lr
        self.weight_decay = weight_decay

        self.train_acc = BinaryAccuracy()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.train_auc = BinaryAUROC()

        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_auc = BinaryAUROC()

        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_auc = BinaryAUROC()

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if X.ndim == 1:
                X = X.unsqueeze(0)
            out = self.model(X)
            return out.cpu().numpy().flatten()

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.float32)

    def _shared_step(self, batch):
        X, y = batch
        y = y.float().unsqueeze(1)
        if X.ndim == 1:
            X = X.unsqueeze(0)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        return pred, y, loss

    def training_step(self, batch, batch_idx):
        pred, y, loss = self._shared_step(batch)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(pred, y))
        self.log("train_precision", self.train_precision(pred, y))
        self.log("train_recall", self.train_recall(pred, y))
        self.log("train_f1", self.train_f1(pred, y))
        self.log("train_auc", self.train_auc(pred, y), prog_bar=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("LR", lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, y, loss = self._shared_step(batch)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc(pred, y))
        self.log("val_precision", self.val_precision(pred, y))
        self.log("val_recall", self.val_recall(pred, y))
        self.log("val_f1", self.val_f1(pred, y))
        self.log("val_auc", self.val_auc(pred, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred, y, loss = self._shared_step(batch)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc(pred, y))
        self.log("test_precision", self.test_precision(pred, y))
        self.log("test_recall", self.test_recall(pred, y))
        self.log("test_f1", self.test_f1(pred, y))
        self.log("test_auc", self.test_auc(pred, y))
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)



