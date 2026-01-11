"""
Latent Diffusion Model Training Script

This script trains a Latent Diffusion Model on unconditional image generation.
"""

import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from controlnet_dataset import LatentDiffusionDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from share import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='./models/ldm_v15.yaml',
                        help='Path to model config file')
    parser.add_argument('--resume_path', type=str, required=True,
                        help='Path to pretrained model checkpoint (e.g., SD v1.5)')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data directory containing images')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./ldm_checkpoints/',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default='ldm',
                        help='Prefix for checkpoint filenames')
    parser.add_argument('--save_every_n_epochs', type=int, default=100,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--save_top_k', type=int, default=1,
                        help='Save top K checkpoints')
    
    # Hardware arguments
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16, choices=[16, 32],
                        help='Training precision (16 for mixed precision, 32 for full)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Accumulate gradients over N batches')
    
    # Logging arguments
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Image logging frequency (in batches)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create model
    print(f'Creating model from config: {args.config}')
    model = create_model(config_path=args.config).cpu()
    
    # Load pretrained weights
    print(f'Loading pretrained weights from: {args.resume_path}')
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'), strict=False)
    print('Pretrained weights loaded successfully')
    
    # Set learning rate
    model.learning_rate = args.learning_rate
    
    # Create dataset and dataloader
    print(f'Loading dataset from: {args.data_path}')
    dataset = LatentDiffusionDataset(path=args.data_path)
    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )
    print(f'Dataset size: {len(dataset)} samples')
    
    # Setup callbacks
    logger = ImageLogger(batch_frequency=args.log_freq)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.checkpoint_prefix}-{{epoch:05d}}-{{loss:.4f}}',
        verbose=True,
        save_top_k=args.save_top_k,
        every_n_epochs=args.save_every_n_epochs,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        precision=args.precision,
        callbacks=[logger, checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs
    )
    
    # Train
    print('Starting training...')
    trainer.fit(model, dataloader)
    print('Training completed!')


if __name__ == '__main__':
    main()
