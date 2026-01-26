"""
UniVG Segmentation Evaluation Script

Evaluate trained UNet model on vascular image segmentation.

Usage:
    python evaluate.py --checkpoint /path/to/model.pth \
                       --test_data_path /path/to/test/data \
                       --output_dir ./results
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from models.unet import UNet
from datasets.vascular_dataset import VascularDataset
from utils.metrics import dice_coef, iou_score, clDice, normalized_surface_dice


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate UNet for vascular segmentation')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to test data')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save predictions and results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predicted masks to output_dir')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    
    # Model parameters (should match training)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--batch_norm', action='store_true')
    
    return parser.parse_args()


def evaluate(args):
    """Main evaluation function."""
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.save_predictions:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    test_dataset = VascularDataset(
        root=args.test_data_path,
        is_train=False,
        is_generated=False,
        transform=None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    model = UNet(
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        padding=True,
        batch_norm=args.batch_norm
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Metrics storage
    metrics = {
        'dice': [],
        'iou': [],
        'cldice': [],
        'nsd': []
    }
    
    # Evaluation loop
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            # Preprocess
            images = images.squeeze(1).permute(0, 3, 1, 2)
            images = images.to(device, dtype=torch.float)
            masks = masks.squeeze(1).cpu().numpy()
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)[:, 1, :, :]  # Class 1 probability
            preds = (probs > 0.5).float().cpu().numpy()
            
            # Calculate metrics for each sample in batch
            for j in range(preds.shape[0]):
                pred = preds[j]
                mask = masks[j]
                
                metrics['dice'].append(dice_coef(pred, mask))
                metrics['iou'].append(iou_score(pred, mask))
                metrics['cldice'].append(clDice(pred, mask))
                metrics['nsd'].append(normalized_surface_dice(pred, mask))
                
                # Save prediction
                if args.save_predictions:
                    pred_img = (pred * 255).astype(np.uint8)
                    pred_pil = Image.fromarray(pred_img)
                    pred_path = os.path.join(args.output_dir, f'pred_{i * args.batch_size + j:04d}.png')
                    pred_pil.save(pred_path)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Dice:   {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}")
    print(f"IoU:    {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}")
    print(f"clDice: {np.mean(metrics['cldice']):.4f} ± {np.std(metrics['cldice']):.4f}")
    print(f"NSD:    {np.mean(metrics['nsd']):.4f} ± {np.std(metrics['nsd']):.4f}")
    print("=" * 50)
    
    # Save results to file
    if args.save_predictions:
        results_path = os.path.join(args.output_dir, 'metrics.txt')
        with open(results_path, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Test data: {args.test_data_path}\n")
            f.write(f"Samples: {len(test_dataset)}\n\n")
            f.write(f"Dice:   {np.mean(metrics['dice']):.4f} ± {np.std(metrics['dice']):.4f}\n")
            f.write(f"IoU:    {np.mean(metrics['iou']):.4f} ± {np.std(metrics['iou']):.4f}\n")
            f.write(f"clDice: {np.mean(metrics['cldice']):.4f} ± {np.std(metrics['cldice']):.4f}\n")
            f.write(f"NSD:    {np.mean(metrics['nsd']):.4f} ± {np.std(metrics['nsd']):.4f}\n")
        print(f"\nResults saved to: {results_path}")
    
    return metrics


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
