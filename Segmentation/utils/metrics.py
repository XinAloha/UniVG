"""
Evaluation metrics for vascular segmentation.

Includes:
- Dice coefficient
- IoU (Intersection over Union)
- clDice (centerline Dice)
- Normalized Surface Dice (NSD)
"""

import numpy as np
import torch
from skimage.morphology import skeletonize, skeletonize_3d
from scipy.ndimage import distance_transform_edt


def dice_coef(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predicted segmentation (numpy array or tensor)
        target: Ground truth segmentation
        smooth: Smoothing factor to prevent division by zero
        
    Returns:
        Dice coefficient value
    """
    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    pred = (pred > 0.5).astype(np.float32)
    target = (target > 0.5).astype(np.float32)
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-5):
    """
    Calculate Intersection over Union (IoU / Jaccard Index).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    if torch.is_tensor(pred):
        pred = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    pred = pred > 0.5
    target = target > 0.5
    
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    
    return (intersection + smooth) / (union + smooth)


def _cl_score(v, s):
    """
    Compute skeleton volume overlap.
    
    Args:
        v: Binary image
        s: Skeleton
        
    Returns:
        Skeleton volume intersection score
    """
    if s.sum() == 0:
        return 0.0
    return np.sum(v * s) / np.sum(s)


def clDice(pred, target):
    """
    Calculate centerline Dice (clDice) metric.
    
    This metric is particularly suited for tubular structures like vessels.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        
    Returns:
        clDice value
    """
    pred = (pred > 0.5).astype(bool)
    target = (target > 0.5).astype(bool)
    
    if len(pred.shape) == 2:
        tprec = _cl_score(pred, skeletonize(target))
        tsens = _cl_score(target, skeletonize(pred))
    elif len(pred.shape) == 3:
        tprec = _cl_score(pred, skeletonize_3d(target))
        tsens = _cl_score(target, skeletonize_3d(pred))
    else:
        raise ValueError(f"Unsupported shape: {pred.shape}")
    
    if tprec + tsens == 0:
        return 0.0
    
    return 2 * tprec * tsens / (tprec + tsens)


def _compute_surface_pixels(segmentation):
    """
    Calculate surface pixels of a binary segmentation.
    
    Args:
        segmentation: Binary segmentation mask
        
    Returns:
        Surface pixel mask
    """
    dt = distance_transform_edt(segmentation)
    return np.logical_and(dt > 0, dt < 1.5)


def normalized_surface_dice(pred, target, tolerance=2.0):
    """
    Calculate Normalized Surface Dice (NSD).
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        tolerance: Distance threshold for surface overlap
        
    Returns:
        NSD value
    """
    pred = (pred > 0.5).astype(bool)
    target = (target > 0.5).astype(bool)
    
    surface_pred = _compute_surface_pixels(pred)
    surface_target = _compute_surface_pixels(target)
    
    dt_pred = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)
    
    overlap1 = np.sum(surface_pred & (dt_target <= tolerance))
    overlap2 = np.sum(surface_target & (dt_pred <= tolerance))
    
    total_surface = np.sum(surface_pred) + np.sum(surface_target)
    
    if total_surface == 0:
        return 1.0
    
    return (overlap1 + overlap2) / total_surface


if __name__ == "__main__":
    # Test metrics
    pred = np.random.rand(256, 256) > 0.5
    target = np.random.rand(256, 256) > 0.5
    
    print(f"Dice: {dice_coef(pred, target):.4f}")
    print(f"IoU: {iou_score(pred, target):.4f}")
    print(f"clDice: {clDice(pred, target):.4f}")
    print(f"NSD: {normalized_surface_dice(pred, target):.4f}")
