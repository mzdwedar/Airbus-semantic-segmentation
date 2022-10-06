from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Dice
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLab(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # model  
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        deeplab = deeplabv3_mobilenet_v3_large(weights=weights, progress=True)
        deeplab.aux_classifier[4] = nn.Conv2d(10, 1, kernel_size=(1, 1), stride=(1, 1))
        deeplab.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        self.model = nn.Sequential(deeplab,
                                   nn.Sigmoid()
                                  )

        # metrics
        self.acc_val = Accuracy(average=None, ignore_index=0)
        self.dice_val = Dice(average=None, ignore_index=0)
        self.f1score_val = F1Score(average=None, ignore_index=0)

        self.acc_test = Accuracy(average=None, ignore_index=0)
        self.f1score_test = F1Score(average=None, ignore_index=0)
        self.dice_test = Dice(average=None, ignore_index=0)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        preds = torch.gt(logits, 0.5)

        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        logits = self.model(x)
        loss = F.BCELoss(logits, y)

        return loss
    
    def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self.model(x)
      loss = F.BCELoss(logits, y)

      preds = torch.gt(logits, 0.5)
      self.acc_val.update(preds, y)
      self.f1score_val.update(preds, y)
      self.dice_val.update(preds, y)

      
      self.log("val_loss", loss, prog_bar=True)
      self.log("acc_val", self.acc_val, prog_bar=True)
      self.log("f1score_val", self.f1score_val, prog_bar=True)
      self.log("dice_val", self.dice_val, prog_nar=True)
    
    def test_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = F.BCELoss(logits, y)

      preds = torch.gt(logits, 0.5)
      self.acc_test.update(preds, y)
      self.f1score_test.update(preds, y)
      self.dice_test.update(preds, y)

      
      self.log("test_loss", loss, prog_bar=True)
      self.log("acc_test", self.acc_test, prog_bar=True)
      self.log("f1score_test", self.f1score_test, prog_bar=True)
      self.log("dice_test", self.dice_test, prog_nar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    