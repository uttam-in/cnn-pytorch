# Crop Disease Diagnosis System

An AI-powered platform for identifying diseases in rice and pulse crops through image analysis.

## Project Structure

```
.
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
├── src/                    # Source code modules
│   └── __init__.py
├── tests/                  # Test suite
│   └── __init__.py
├── models/                 # Trained model weights
├── data/                   # Test images and datasets
└── config/                 # Configuration files
    ├── model_config.json   # Model parameters and paths
    └── class_names.json    # Disease class labels
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run app.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/
```

## Configuration

- `config/model_config.json`: Contains model architecture parameters, preprocessing settings, and inference configuration
- `config/class_names.json`: Lists the disease classes the model can identify

## Development

This project follows a modular architecture with clear separation between:
- Authentication layer
- Image processing pipeline
- CNN model inference
- Streamlit user interface

See the design document in `.kiro/specs/crop-disease-diagnosis/design.md` for detailed architecture information.

pytest:

python -m pytest tests/test_transforms.py -v