"""
ControlNet Training Script with Pretrained Weights

This script trains a ControlNet model by loading pretrained weights from a 
Latent Diffusion Model checkpoint and fine-tuning on paired image data.
"""

import gc
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from controlnet_dataset import ControlNetDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
from share import *


def get_node_name(name, parent_name):
    """Extract node name by removing parent prefix."""
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


def load_pretrained_weights(model, pretrained_path):
    """Load pretrained weights from LDM checkpoint into ControlNet model."""
    print(f'Loading pretrained weights from: {pretrained_path}')
    pretrained_weights = torch.load(pretrained_path)
    
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']

    scratch_dict = model.state_dict()
    target_dict = {}
    
    for k in scratch_dict.keys():
        is_control, name = get_node_name(k, 'control_')
        if is_control:
            copy_k = 'model.diffusion_' + name
        else:
            copy_k = k
        
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'Newly initialized weight: {k}')
    
    print('Pretrained weights loaded successfully.')
    return target_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Train ControlNet with pretrained weights')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='./models/cldm_v15.yaml',
                        help='Path to model config file')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained LDM checkpoint')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data directory (should contain source/ and target/ subdirs)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=2000,
                        help='Maximum number of training epochs')
    parser.add_argument('--sd_locked', action='store_true', default=True,
                        help='Lock Stable Diffusion weights during training')
    parser.add_argument('--only_mid_control', action='store_false', default=False,
                        help='Only use middle control')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./control_checkpoints/',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_prefix', type=str, default='controlnet',
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
    parser.add_argument('--log_freq', type=int, default=1000000000,
                        help='Image logging frequency (in steps)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create model
    print(f'Creating model from config: {args.config}')
    model = create_model(config_path=args.config)
    
    # Load pretrained weights
    target_dict = load_pretrained_weights(model, args.pretrained)
    model = model.cpu()
    model.load_state_dict(target_dict, strict=True)
    
    # Clean up memory
    del target_dict
    gc.collect()
    
    # Set model parameters
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    
    # Create dataset and dataloader
    print(f'Loading dataset from: {args.data_path}')
    dataset = ControlNetDataset(path=args.data_path)
    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True
    )
    print(f'Dataset size: {len(dataset)} samples')
    
    # Setup checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f'{args.checkpoint_prefix}-{{step:05d}}-{{loss:.4f}}',
        verbose=True,
        save_top_k=args.save_top_k,
        every_n_epochs=args.save_every_n_epochs,
    )
    
    # Create trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
    )
    
    # Train
    print('Starting training...')
    trainer.fit(model, dataloader)
    print('Training completed!')


if __name__ == '__main__':
    main()
