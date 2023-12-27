"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.

"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_data_loaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int= NUM_WORKERS
):
      
      """Creates training and testing DataLoaders.
  """
      
    # Use ImageFolder to create dataset(s)
      train_data = datasets.ImageFolder(train_dir, transform=transform)
      test_data = datasets.ImageFolder(test_dir, transform=transform)

      class_names = train_data.classes

      #turn images into data loaders

      train_dataloader = DataLoader(
            train_data,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
      )

      test_dataloader = DataLoader(
            test_data,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
      )

      return train_dataloader, test_dataloader, class_names

