"""
Script to prepare test data from the rice leaf diseases dataset.

This script copies a few sample images from each disease class to a
test_images directory for manual testing of the application.
"""

import os
import shutil
from pathlib import Path


def prepare_test_data():
    """Copy sample images from dataset to test directory."""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / 'data' / 'rice_leaf_diseases'
    test_dir = base_dir / 'data' / 'test_images'
    
    # Create test directory
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Preparing test data from: {source_dir}")
    print(f"Copying to: {test_dir}")
    
    # Get disease classes from directory
    disease_classes = [d for d in source_dir.iterdir() if d.is_dir()]
    
    if not disease_classes:
        print("Warning: No disease class directories found!")
        return
    
    print(f"\nFound {len(disease_classes)} disease classes:")
    for disease in disease_classes:
        print(f"  - {disease.name}")
    
    # Copy 2-3 sample images from each class
    samples_per_class = 2
    total_copied = 0
    
    print(f"\nCopying {samples_per_class} sample(s) from each class...")
    
    for disease_dir in disease_classes:
        # Get all image files
        image_files = list(disease_dir.glob('*.jpg')) + list(disease_dir.glob('*.JPG'))
        
        if not image_files:
            print(f"  Warning: No images found in {disease_dir.name}")
            continue
        
        # Copy first N images
        for i, img_file in enumerate(image_files[:samples_per_class]):
            # Create descriptive filename
            new_name = f"{disease_dir.name.replace(' ', '_')}_{i+1}{img_file.suffix}"
            dest_path = test_dir / new_name
            
            shutil.copy2(img_file, dest_path)
            total_copied += 1
            print(f"  ✓ Copied: {new_name}")
    
    print(f"\n✓ Successfully copied {total_copied} test images to {test_dir}")
    
    # Create a README in test_images
    readme_path = test_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("# Test Images\n\n")
        f.write("This directory contains sample images for testing the Crop Disease Diagnosis System.\n\n")
        f.write("## Disease Classes\n\n")
        for disease_dir in sorted(disease_classes):
            count = len([f for f in test_dir.glob(f"{disease_dir.name.replace(' ', '_')}*")])
            f.write(f"- **{disease_dir.name}**: {count} sample(s)\n")
        f.write("\n## Usage\n\n")
        f.write("Upload these images through the Streamlit interface to test disease prediction.\n")
    
    print(f"✓ Created README at {readme_path}")


if __name__ == "__main__":
    prepare_test_data()
