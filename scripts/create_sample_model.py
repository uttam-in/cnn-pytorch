"""
Script to create a sample trained model for testing purposes.

This script initializes the CropDiseaseClassifier model with random weights
and saves it to the models directory. This allows testing of the inference
pipeline without requiring a fully trained model.
"""

import sys
import os
import json
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import CropDiseaseClassifier


def create_sample_model():
    """Create and save a sample model with random initialization."""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_classes = config['model']['num_classes']
    model_path = os.path.join(os.path.dirname(__file__), '..', config['model']['model_path'])
    
    print(f"Creating sample model with {num_classes} classes...")
    
    # Initialize model
    model = CropDiseaseClassifier(num_classes=num_classes)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    print(f"Sample model saved to: {model_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loading the model
    print("\nTesting model loading...")
    test_model = CropDiseaseClassifier(num_classes=num_classes)
    test_model.load_state_dict(torch.load(model_path))
    test_model.eval()
    
    # Test forward pass
    print("Testing forward pass...")
    test_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = test_model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be ~1.0): {output.sum().item():.6f}")
    print(f"Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    
    print("\n✓ Sample model created and tested successfully!")


if __name__ == "__main__":
    create_sample_model()
