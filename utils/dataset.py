import os

from typing import Optional

import numpy as np

from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms as T
import pytorch_lightning as pl


def rle_to_pixels(rle_code):
  """
  Decode the segmentation mask from run-length-encoding
  1.convert the string into tokens that represents start and length
  2. unravel the the pixels range(start, start+length)
  3. map the pixel to 2D, whose shape is 768*768
  """
  rle_code = [int(i) for i in rle_code.split()]
  pixels = [(pixel_position % 768, pixel_position // 768) 
                for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
                for pixel_position in range(start, start + length)]
  return pixels

def pixels_to_mask(pixels):
  """
  project the pixels onto a canvas of 768*768

  1. create a sparse tensor with the decoded pixels
  2. change to dense tensor
  3. add a dimension -> to make the dimensions: (768,768,1)
  """
  canvas = np.zeros((768, 768))

  canvas[tuple(zip(*pixels))] = 1

  return torch.as_tensor(np.expand_dims(canvas, axis=0), dtype=torch.float32)

class AirbusCustomDataset(Dataset):
  """
  create a custom dataset
  
  Args:
    images_dir: the path that contains all images
    annotations: a dataframe, where each record(filename, Run-length encoding)
    transform: transformations on input images (resizing, normalization, augmentation, etc)
    target_transform: transform run-length encoding to segmentation mask
  
  returns:
    Dataset object
  """
  def __init__(self, images_dir, annotations, transform=None, target_transform=None):
    self.annotations = annotations
    self.images_dir = images_dir
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, idx):
    img_path = os.path.join(self.images_dir, self.annotations.iloc[idx, 0])
    image = Image.open(img_path)
    segmentation = self.annotations.iloc[idx, 1]

    if(self.transform):
      image = self.transform(image)

    if(self.target_transform):
      segmentation = self.target_transform(segmentation)

    return image, segmentation

class AirbusDataModule(pl.LightningDataModule):
  def __init__(self, data_dir, annotations, batch_size):
    super().__init__()
    
    self.data_dir = data_dir
    self.annotations = annotations
    self.batch_size = batch_size

    self.transform = T.Compose([T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST),
                                # transforms.RandomResizedCrop(224),
                                # transforms.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
    self.target_transform = T.Compose([rle_to_pixels,
                                      pixels_to_mask,
                                      T.Resize((128, 128), interpolation=T.InterpolationMode.NEAREST)        
                                     ])

  def setup(self, stage: Optional[str] = None):
    # Assign train/val datasets for use in dataloaders

    if stage in ("fit", None):
      n = len(self.annotations)
      self.train_ds = AirbusCustomDataset(self.data_dir, self.annotations[:int(.8*n)], transform=self.transform, target_transform=self.target_transform)
      self.val_ds = AirbusCustomDataset(self.data_dir, self.annotations[int(.8*n):int(.9*n)], transform=self.transform, target_transform=self.target_transform)
    
    # Assign test dataset for use in dataloader(s)
    if stage in ("test", None):
      n = len(self.annotations)
      self.test_ds = AirbusCustomDataset(self.data_dir, self.annotations[int(.9*n):], transform=self.transform, target_transform=self.target_transform)
  
  def train_dataloader(self):
    return DataLoader(self.train_ds, batch_size=self.batch_size)
  
  def val_dataloader(self):
    return DataLoader(self.val_ds, batch_size=self.batch_size)
  
  def test_dataloader(self):
    return DataLoader(self.test_ds, batch_size=self.batch_size)

  
