import os 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from utils.dataset import AirbusDataModule
from utils.model import DeepLab
from utils.extract_dataset import extract
import argparse

checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                        save_top_k=1,
                                        every_n_epochs=10,
                                        dirpath='saved_models/',
                                        filename='airbus-deeplab3-{epoch:02d}-{val_loss:.2f}'
                                     )

wandb_logger = WandbLogger(project="airbus-segmentation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    os.makedirs('saved_models', exist_ok=True)
    wandb_logger.experiment.config.update({'model':'deeplab3', 'encoder': 'resnet18', 'encoder_depth':5})

    segments = extract(args.path)
    airbus = AirbusDataModule('./train_v2', segments, args.batch_size)
    model = DeepLab()

    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=args.num_epochs, 
                        logger=wandb_logger, callbacks=[checkpoint_callback])
    trainer.fit(model, airbus)