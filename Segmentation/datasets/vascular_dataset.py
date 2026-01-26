"""
Vascular Image Dataset for Segmentation Training

Supports both real and generated/synthetic vascular images.
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from albumentations import Normalize


class VascularDataset(Dataset):
    """
    Dataset for vascular image segmentation.
    
    Args:
        root: Root directory containing image data
        is_train: If True, load training data; otherwise load test data
        is_generated: If True, load generated/synthetic data structure
        transform: Albumentations transform to apply
        
    Directory structure:
        For real data:
            root/
            ├── train_origin/   # Training images
            ├── train_mask/     # Training masks
            ├── test_origin/    # Test images
            └── test_mask/      # Test masks
        
        For generated data:
            root/
            ├── images/         # Generated images
            └── masks/          # Corresponding masks
    """
    
    def __init__(self, root, is_train=True, is_generated=False, transform=None):
        self.root = root
        self.transform = transform
        self.is_generated = is_generated
        
        if is_generated:
            self.image_dir = os.path.join(root, "images")
            self.mask_dir = os.path.join(root, "masks")
        else:
            if is_train:
                self.image_dir = os.path.join(root, "train_origin")
                self.mask_dir = os.path.join(root, "train_mask")
            else:
                self.image_dir = os.path.join(root, "test_origin")
                self.mask_dir = os.path.join(root, "test_mask")
        
        # Get sorted file lists
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        
        assert len(self.images) == len(self.masks), \
            f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = np.array(image)
        mask = np.array(mask)
        
        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # Normalize image to [-1, 1]
        image = Normalize(mean=0.5, std=0.5)(image=image)["image"]
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask


if __name__ == "__main__":
    # Test dataset
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    # Replace with your data path
    dataset = VascularDataset(
        root="./test_data",
        is_train=True,
        is_generated=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
