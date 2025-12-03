"""
Image transformation pipeline for crop disease diagnosis.

This module provides the ImageTransformer class that handles all image
preprocessing operations including resizing, normalization, and tensor conversion.
"""

from typing import Tuple, List, Union
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


class ImageTransformer:
    """
    Handles image transformations for both training and inference.
    
    This class provides a consistent transformation pipeline that preprocesses
    images for the CNN model, including resizing, normalization, and conversion
    to PyTorch tensors.
    
    Attributes:
        target_size: Target dimensions (height, width) for resized images
        mean: Mean values for normalization (one per channel)
        std: Standard deviation values for normalization (one per channel)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Initialize the ImageTransformer.
        
        Args:
            target_size: Target dimensions (height, width) for resizing
            mean: Mean values for normalization (RGB channels)
            std: Standard deviation values for normalization (RGB channels)
        """
        self.target_size = target_size
        self.mean = mean
        self.std = std
        
        # Create transformation pipelines
        self._inference_transforms = self._build_inference_transforms()
        self._training_transforms = self._build_training_transforms()
    
    def _build_inference_transforms(self) -> transforms.Compose:
        """
        Build the inference transformation pipeline.
        
        Returns:
            Composed transformation pipeline for inference
        """
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def _build_training_transforms(self) -> transforms.Compose:
        """
        Build the training transformation pipeline with augmentations.
        
        Returns:
            Composed transformation pipeline for training with augmentations
        """
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def transform(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Apply inference transformations to an image.
        
        This method applies the standard inference pipeline: resize, convert to
        tensor, and normalize. It handles both PIL Images and NumPy arrays.
        
        Args:
            image: Input image as PIL Image or NumPy array
            
        Returns:
            Transformed image as PyTorch tensor with shape (C, H, W)
            
        Raises:
            TypeError: If image is not a PIL Image or NumPy array
            ValueError: If image cannot be processed
        """
        # Convert NumPy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            # Handle different array shapes
            if image.ndim == 2:  # Grayscale
                image = Image.fromarray(image, mode='L').convert('RGB')
            elif image.ndim == 3:
                if image.shape[2] == 3:  # RGB
                    image = Image.fromarray(image.astype('uint8'), mode='RGB')
                elif image.shape[2] == 4:  # RGBA
                    image = Image.fromarray(image.astype('uint8'), mode='RGBA').convert('RGB')
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unsupported array dimensions: {image.ndim}")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Image must be PIL Image or NumPy array, got {type(image)}")
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply inference transformations
        return self._inference_transforms(image)
    
    def get_inference_transforms(self) -> transforms.Compose:
        """
        Get the inference transformation pipeline.
        
        Returns:
            Composed transformation pipeline for inference
        """
        return self._inference_transforms
    
    def get_training_transforms(self) -> transforms.Compose:
        """
        Get the training transformation pipeline with augmentations.
        
        Returns:
            Composed transformation pipeline for training
        """
        return self._training_transforms
    
    def resize(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Resize an image to target dimensions.
        
        Args:
            image: Input image as PIL Image or NumPy array
            
        Returns:
            Resized PIL Image
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image, mode='L').convert('RGB')
            else:
                image = Image.fromarray(image.astype('uint8'), mode='RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Image must be PIL Image or NumPy array, got {type(image)}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return transforms.Resize(self.target_size)(image)
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize a tensor using configured mean and std.
        
        Args:
            tensor: Input tensor with shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Normalized tensor
        """
        return transforms.Normalize(mean=self.mean, std=self.std)(tensor)
    
    def to_tensor(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Convert PIL Image or NumPy array to PyTorch tensor.
        
        Args:
            image: Input image as PIL Image or NumPy array
            
        Returns:
            Image as PyTorch tensor with shape (C, H, W)
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image, mode='L').convert('RGB')
            else:
                image = Image.fromarray(image.astype('uint8'), mode='RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Image must be PIL Image or NumPy array, got {type(image)}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return transforms.ToTensor()(image)
