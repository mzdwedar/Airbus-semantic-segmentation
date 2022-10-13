import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Dice
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DeepLab(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters()

        # model  
        self.model = smp.DeepLabV3(encoder_name='resnet18', 
                                    encoder_depth=5, 
                                    encoder_weights='imagenet', 
                                    classes=1,
                                    activation='sigmoid',
                                    in_channels=3
                                  )

        # metrics
        self.acc_val = Accuracy(ignore_index=0, mdmc_reduce='samplewise')
        self.dice_val = Dice(ignore_index=0, mdmc_reduce='samplewise')
        self.f1score_val = F1Score(ignore_index=0, mdmc_reduce='samplewise')

        self.acc_test = Accuracy(ignore_index=0, mdmc_reduce='samplewise')
        self.f1score_test = F1Score(ignore_index=0, mdmc_reduce='samplewise')
        self.dice_test = Dice(ignore_index=0, mdmc_reduce='samplewise')

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        preds = torch.gt(logits, 0.5)

        return preds

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        logits = self.model(x)
        loss = F.binary_cross_entropy(logits, y)

        return loss
    
    def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self.model(x)
      loss = F.binary_cross_entropy(logits, y)

      preds = torch.gt(logits, 0.5)

      y = y.to(torch.int32)
      self.acc_val.update(preds, y)
      self.f1score_val.update(preds, y)
      self.dice_val.update(preds, y)

      
      self.log("val_loss", loss, prog_bar=True)
      self.log("acc_val", self.acc_val, prog_bar=True)
      self.log("f1score_val", self.f1score_val, prog_bar=True)
      self.log("dice_val", self.dice_val, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = F.binary_cross_entropy(logits, y)

      preds = torch.gt(logits, 0.5)

      y = y.to(torch.int32)
      
      self.acc_test.update(preds, y)
      self.f1score_test.update(preds, y)
      self.dice_test.update(preds, y)

      
      self.log("test_loss", loss, prog_bar=True)
      self.log("acc_test", self.acc_test, prog_bar=True)
      self.log("f1score_test", self.f1score_test, prog_bar=True)
      self.log("dice_test", self.dice_test, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    