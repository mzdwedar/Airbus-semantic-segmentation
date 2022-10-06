from pytorch_lightning import Trainer
import torch

from utils.dataset import AirbusDataModule
from utils.model import DeepLab
from utils.extract_dataset import extract
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()

    segments = extract(args.path)
    airbus = AirbusDataModule('train_v2', segments, args.batch_size)
    model = DeepLab()

    trainer = Trainer(max_epochs=args.num_epochs)
    trainer.fit(model, airbus)