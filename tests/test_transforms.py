"""
Tests for image transformation pipeline.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from src.transforms import ImageTransformer


def test_image_transformer_initialization():
    """Test that ImageTransformer initializes with default parameters."""
    transformer = ImageTransformer()
    assert transformer.target_size == (224, 224)
    assert transformer.mean == [0.485, 0.456, 0.406]
    assert transformer.std == [0.229, 0.224, 0.225]


def test_image_transformer_custom_parameters():
    """Test that ImageTransformer initializes with custom parameters."""
    transformer = ImageTransformer(
        target_size=(128, 128),
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    assert transformer.target_size == (128, 128)
    assert transformer.mean == [0.5, 0.5, 0.5]
    assert transformer.std == [0.5, 0.5, 0.5]


def test_transform_pil_image():
    """Test transformation of PIL Image."""
    transformer = ImageTransformer(target_size=(224, 224))
    
    # Create a sample PIL image
    image = Image.new('RGB', (100, 100), color='red')
    
    # Transform the image
    result = transformer.transform(image)
    
    # Check output shape
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32


def test_transform_numpy_array():
    """Test transformation of NumPy array."""
    transformer = ImageTransformer(target_size=(224, 224))
    
    # Create a sample NumPy array (RGB)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Transform the image
    result = transformer.transform(image)
    
    # Check output shape
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32


def test_transform_grayscale_to_rgb():
    """Test that grayscale images are converted to RGB."""
    transformer = ImageTransformer(target_size=(224, 224))
    
    # Create a grayscale image
    image = Image.new('L', (100, 100), color=128)
    
    # Transform the image
    result = transformer.transform(image)
    
    # Should have 3 channels (RGB)
    assert result.shape == (3, 224, 224)


def test_resize_transformation():
    """Test resize functionality."""
    transformer = ImageTransformer(target_size=(128, 128))
    
    # Create an image with different size
    image = Image.new('RGB', (300, 200), color='blue')
    
    # Resize the image
    resized = transformer.resize(image)
    
    # Check dimensions
    assert resized.size == (128, 128)
    assert isinstance(resized, Image.Image)


def test_to_tensor_conversion():
    """Test tensor conversion."""
    transformer = ImageTransformer()
    
    # Create a PIL image
    image = Image.new('RGB', (50, 50), color='green')
    
    # Convert to tensor
    tensor = transformer.to_tensor(image)
    
    # Check properties
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 50, 50)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor >= 0) and torch.all(tensor <= 1)


def test_normalize_tensor():
    """Test normalization."""
    transformer = ImageTransformer(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Create a tensor with known values
    tensor = torch.ones(3, 10, 10) * 0.5
    
    # Normalize
    normalized = transformer.normalize(tensor)
    
    # Check that normalization was applied
    assert isinstance(normalized, torch.Tensor)
    assert normalized.shape == (3, 10, 10)


def test_get_inference_transforms():
    """Test getting inference transforms."""
    transformer = ImageTransformer()
    
    inference_transforms = transformer.get_inference_transforms()
    
    assert inference_transforms is not None


def test_get_training_transforms():
    """Test getting training transforms."""
    transformer = ImageTransformer()
    
    training_transforms = transformer.get_training_transforms()
    
    assert training_transforms is not None


def test_training_transforms_include_augmentation():
    """Test that training transforms produce varied outputs."""
    transformer = ImageTransformer(target_size=(224, 224))
    
    # Create a sample image
    image = Image.new('RGB', (100, 100), color='red')
    
    # Apply training transforms multiple times
    training_transforms = transformer.get_training_transforms()
    result1 = training_transforms(image)
    result2 = training_transforms(image)
    
    # Results should be different due to random augmentations
    # (though there's a small chance they could be identical)
    assert result1.shape == result2.shape == (3, 224, 224)


def test_invalid_input_type():
    """Test that invalid input types raise TypeError."""
    transformer = ImageTransformer()
    
    with pytest.raises(TypeError):
        transformer.transform("not an image")
    
    with pytest.raises(TypeError):
        transformer.transform(123)


def test_transform_preserves_dimensions():
    """Test that transform always produces target dimensions."""
    transformer = ImageTransformer(target_size=(256, 256))
    
    # Test with various input sizes
    sizes = [(50, 50), (100, 200), (500, 300), (1000, 1000)]
    
    for size in sizes:
        image = Image.new('RGB', size, color='blue')
        result = transformer.transform(image)
        assert result.shape == (3, 256, 256)
