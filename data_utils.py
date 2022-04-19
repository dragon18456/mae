import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image
import random
import numpy as np

def find_classes(dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

class ValidationDataSet(Dataset):
    def __init__(self, main_dir, transform, class_to_dict):
        self.main_dir = main_dir
        self.transform = transform
        image_dir = os.path.join(main_dir, 'images')
        all_imgs = os.listdir(image_dir)
        self.total_imgs = all_imgs
        self.map = {}
        with open(os.path.join(self.main_dir, 'val_annotations.txt')) as f:
            for line in f:
                values = line.split()
                key, value = values[0], values[1]
                self.map[key] = class_to_dict[value]
        self.main_dir = image_dir

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        target = self.map[self.total_imgs[idx]]
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, target

def loadData(data_dir, train_batch_size, val_batch_size):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    _, class_to_dict = find_classes(train_dir)

    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    #val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_dataset = ValidationDataSet(val_dir, transform=val_transform, class_to_dict=class_to_dict)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=2)

    return train_loader, val_dataset


