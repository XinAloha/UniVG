"""
Dataset classes for ControlNet and Latent Diffusion Model training.
"""

import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from albumentations import Normalize


class ControlNetDataset(Dataset):
    """
    Dataset for ControlNet training with paired source (condition) and target images.
    
    Args:
        path (str): Root directory containing 'source' and 'target' subdirectories
    
    Directory structure:
        path/
        ├── source/  # Condition images (masks, edges, etc.)
        │   ├── image_001.png
        │   └── ...
        └── target/  # Target images (ground truth)
            ├── image_001.png
            └── ...
    """
    def __init__(self, path):
        self.path = path
        self.target_path = os.path.join(self.path, 'target')
        self.source_path = os.path.join(self.path, 'source')
        self.target = os.listdir(self.target_path)
        self.source = os.listdir(self.source_path)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        source_path = os.path.join(self.source_path, self.source[idx])
        target_path = os.path.join(self.target_path, self.target[idx])
        prompt = ""

        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # Convert from BGR to RGB (OpenCV default is BGR)
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1]
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1]
        target = Normalize(mean=0.5, std=0.5)(image=target)["image"]
        
        # Generate edge placeholder (for compatibility)
        edges = torch.rand(1, 3, 512, 512)
        
        return dict(jpg=target, txt=prompt, hint=source, edges=edges[0].squeeze(0))


class LatentDiffusionDataset(Dataset):
    """
    Dataset for Latent Diffusion Model training (unconditional generation).
    
    Args:
        path (str): Directory containing target images
    """
    def __init__(self, path):
        self.path = path
        self.target = os.listdir(self.path)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        target_path = os.path.join(self.path, self.target[idx])
        prompt = ""
        target = cv2.imread(target_path)

        # Convert from BGR to RGB
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        # Normalize target images to [-1, 1]
        target = Normalize(mean=0.5, std=0.5)(image=target)["image"]

        return dict(jpg=target, txt=prompt)
