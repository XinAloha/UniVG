"""
UniVG Segmentation Training Script

Train UNet model with joint real and generated data for vascular image segmentation.

Usage:
    python train.py --real_data_path /path/to/real/data \
                    --fake_data_path /path/to/generated/data \
                    --output_dir ./checkpoints \
                    --batch_size 2 \
                    --steps 75000
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.unet import UNet
from datasets.vascular_dataset import VascularDataset
from utils.augmentation import get_train_augmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet for vascular segmentation')
    
    # Data paths
    parser.add_argument('--real_data_path', type=str, required=True,
                        help='Path to real training data')
    parser.add_argument('--fake_data_path', type=str, required=True,
                        help='Path to generated/synthetic training data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--steps', type=int, default=75000,
                        help='Total training steps (default: 75000)')
    parser.add_argument('--save_steps', type=int, default=6250,
                        help='Save checkpoint every N steps (default: 6250)')
    
    # Model parameters
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels (default: 3)')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--batch_norm', action='store_true',
                        help='Use batch normalization')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def generate_joint_batches(real_loader, fake_loader):
    """
    Generator that yields joint batches of real and generated data.
    
    Args:
        real_loader: DataLoader for real data
        fake_loader: DataLoader for generated data
        
    Yields:
        tuple: (images, masks) - concatenated real and fake batches
    """
    while True:
        real_iter = iter(real_loader)
        fake_iter = iter(fake_loader)
        
        for real_batch, fake_batch in zip(real_iter, fake_iter):
            real_image, real_mask = real_batch
            fake_image, fake_mask = fake_batch
            
            images = torch.cat((real_image, fake_image), dim=0)
            masks = torch.cat((real_mask, fake_mask), dim=0)
            
            yield images, masks


def count_parameters(model):
    """Count trainable parameters in millions."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1_000_000


def train(args):
    """Main training function."""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get augmentation
    transform = get_train_augmentation()
    
    # Create datasets
    real_dataset = VascularDataset(
        root=args.real_data_path,
        is_train=True,
        is_generated=False,
        transform=transform
    )
    
    fake_dataset = VascularDataset(
        root=args.fake_data_path,
        is_train=True,
        is_generated=True,
        transform=transform
    )
    
    print(f"Real dataset size: {len(real_dataset)}")
    print(f"Generated dataset size: {len(fake_dataset)}")
    
    # Create dataloaders
    real_loader = DataLoader(
        real_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    fake_loader = DataLoader(
        fake_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = UNet(
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        padding=True,
        batch_norm=args.batch_norm
    )
    model = model.to(device)
    print(f"Model has {count_parameters(model):.2f}M trainable parameters")
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from checkpoint: {args.resume}, step: {start_step}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    data_generator = generate_joint_batches(real_loader, fake_loader)
    
    pbar = tqdm(range(start_step, args.steps), desc='Training')
    for step in pbar:
        model.train()
        
        images, masks = next(data_generator)
        
        # Preprocess
        images = images.squeeze(1).permute(0, 3, 1, 2)  # (B, C, H, W)
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.long).squeeze(1)
        
        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Save checkpoint
        if (step + 1) % args.save_steps == 0:
            checkpoint_path = os.path.join(
                args.output_dir, 
                f'unet_step_{step + 1}.pth'
            )
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"\nSaved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'unet_final.pth')
    torch.save({
        'step': args.steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
