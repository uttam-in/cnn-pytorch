# Setup Scripts

This directory contains utility scripts for setting up the Crop Disease Diagnosis System.

## Scripts

### `create_sample_model.py`

Creates a sample trained model with random initialization for testing purposes.

**Usage:**
```bash
python scripts/create_sample_model.py
```

**What it does:**
- Initializes a `CropDiseaseClassifier` model with the configured number of classes
- Saves the model weights to `models/crop_disease_model.pth`
- Tests that the model can be loaded and performs inference correctly
- Validates output shape and probability distribution

**Output:**
- `models/crop_disease_model.pth` - PyTorch model state dict file

### `prepare_test_data.py`

Prepares test images from the rice leaf diseases dataset for manual testing.

**Usage:**
```bash
python scripts/prepare_test_data.py
```

**What it does:**
- Copies 2 sample images from each disease class in `data/rice_leaf_diseases/`
- Creates a `data/test_images/` directory with organized test samples
- Generates a README documenting the test images
- Renames files with descriptive names (e.g., `Bacterial_leaf_blight_1.jpg`)

**Output:**
- `data/test_images/` - Directory containing 6 test images (2 per disease class)
- `data/test_images/README.md` - Documentation of test images

### `verify_setup.py`

Comprehensive verification script that checks all system components are properly configured.

**Usage:**
```bash
python scripts/verify_setup.py
```

**What it checks:**
1. Configuration files exist and are valid (`config/model_config.json`, `config/class_names.json`)
2. Model file exists and can be loaded (`models/crop_disease_model.pth`)
3. Test images exist and can be read (`data/test_images/`)
4. Image transformation pipeline works correctly
5. End-to-end inference pipeline produces valid predictions

**Exit codes:**
- `0` - All checks passed
- `1` - One or more checks failed

## Quick Start

To set up the system from scratch, run these scripts in order:

```bash
# 1. Create the sample model
python scripts/create_sample_model.py

# 2. Prepare test data
python scripts/prepare_test_data.py

# 3. Verify everything is working
python scripts/verify_setup.py
```

If all scripts complete successfully, the system is ready for testing!

## Requirements

These scripts require the following Python packages:
- torch
- Pillow
- The project's `src` modules (model, transforms)

Install dependencies with:
```bash
pip install -r requirements.txt
```
