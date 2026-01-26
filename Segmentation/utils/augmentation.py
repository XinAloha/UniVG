"""
Data augmentation utilities for vascular image segmentation.
"""

from albumentations import (
    Compose,
    RandomRotate90,
    Flip,
    Transpose,
    OneOf,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    Blur,
    ShiftScaleRotate,
    OpticalDistortion,
    GridDistortion,
    CLAHE,
    RandomBrightnessContrast,
    HueSaturationValue,
)


def get_train_augmentation(p=0.5):
    """
    Get training augmentation pipeline.
    
    Args:
        p: Probability of applying the transform
        
    Returns:
        Albumentations Compose object
    """
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(
            shift_limit=0.0625, 
            scale_limit=0.2, 
            rotate_limit=45, 
            p=0.5
        ),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
        ], p=0.3),
        OneOf([
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def get_test_augmentation():
    """
    Get test/validation augmentation pipeline (no augmentation).
    
    Returns:
        None (no augmentation for testing)
    """
    return None
