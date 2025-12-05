"""
Verification script to ensure model and test data are properly set up.

This script checks that:
1. Model file exists and can be loaded
2. Test images exist and can be read
3. Configuration files are valid
4. Basic inference pipeline works
"""

import sys
import os
import json
import torch
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import CropDiseaseClassifier
from src.transforms import ImageTransformer


def verify_setup():
    """Verify that all components are properly set up."""
    
    base_dir = Path(__file__).parent.parent
    print("=" * 60)
    print("Crop Disease Diagnosis System - Setup Verification")
    print("=" * 60)
    
    # 1. Check configuration files
    print("\n1. Checking configuration files...")
    config_path = base_dir / 'config' / 'model_config.json'
    class_names_path = base_dir / 'config' / 'class_names.json'
    
    if not config_path.exists():
        print("  ✗ model_config.json not found!")
        return False
    print(f"  ✓ model_config.json found")
    
    if not class_names_path.exists():
        print("  ✗ class_names.json not found!")
        return False
    print(f"  ✓ class_names.json found")
    
    # Load configs
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)['classes']
    
    print(f"  ✓ Number of classes: {len(class_names)}")
    print(f"  ✓ Classes: {', '.join(class_names[:3])}...")
    
    # 2. Check model file
    print("\n2. Checking model file...")
    model_path = base_dir / config['model']['model_path']
    
    if not model_path.exists():
        print(f"  ✗ Model file not found at {model_path}")
        return False
    print(f"  ✓ Model file found at {model_path}")
    
    # Load model
    try:
        model = CropDiseaseClassifier(num_classes=config['model']['num_classes'])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False
    
    # 3. Check test images
    print("\n3. Checking test images...")
    test_images_dir = base_dir / 'data' / 'test_images'
    
    if not test_images_dir.exists():
        print(f"  ✗ Test images directory not found")
        return False
    
    image_files = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.JPG'))
    print(f"  ✓ Found {len(image_files)} test images")
    
    if len(image_files) == 0:
        print("  ✗ No test images found!")
        return False
    
    # Try loading a test image
    try:
        test_image = Image.open(image_files[0])
        print(f"  ✓ Successfully loaded test image: {image_files[0].name}")
        print(f"  ✓ Image size: {test_image.size}")
    except Exception as e:
        print(f"  ✗ Failed to load test image: {e}")
        return False
    
    # 4. Test transformation pipeline
    print("\n4. Testing transformation pipeline...")
    try:
        transformer = ImageTransformer(
            target_size=tuple(config['model']['input_size']),
            mean=config['preprocessing']['mean'],
            std=config['preprocessing']['std']
        )
        
        # Transform test image
        transformed = transformer.transform(test_image)
        print(f"  ✓ Image transformed successfully")
        print(f"  ✓ Transformed shape: {transformed.shape}")
        print(f"  ✓ Transformed dtype: {transformed.dtype}")
        print(f"  ✓ Value range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    except Exception as e:
        print(f"  ✗ Transformation failed: {e}")
        return False
    
    # 5. Test inference
    print("\n5. Testing inference pipeline...")
    try:
        # Add batch dimension
        input_tensor = transformed.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Get prediction
        confidence, predicted_idx = torch.max(output, 1)
        predicted_class = class_names[predicted_idx.item()]
        
        print(f"  ✓ Inference completed successfully")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Predicted class: {predicted_class}")
        print(f"  ✓ Confidence: {confidence.item() * 100:.2f}%")
        print(f"  ✓ Output sum: {output.sum().item():.6f} (should be ~1.0)")
        
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 60)
    print("✓ All verification checks passed!")
    print("=" * 60)
    print("\nThe system is ready for testing. You can now:")
    print("  1. Run the Streamlit app: streamlit run app.py")
    print("  2. Run tests: pytest tests/")
    print("  3. Upload test images from: data/test_images/")
    
    return True


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
