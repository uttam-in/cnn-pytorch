# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `models/`, `data/`, `config/`
  - Create `requirements.txt` with PyTorch, Streamlit, Pillow, Hypothesis, pytest
  - Create main application entry point `app.py`
  - Create configuration file for model parameters and paths
  - _Requirements: 6.1_

- [x] 2. Implement CNN model architecture
  - Create `src/model.py` with `CropDiseaseClassifier` class inheriting from `nn.Module`
  - Implement convolutional blocks with Conv2d, ReLU, MaxPool2d, and Dropout layers
  - Implement fully connected layers for classification
  - Implement forward pass method
  - Add model initialization with configurable number of classes
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ]* 2.1 Write unit tests for model architecture
  - Test model instantiation with different numbers of classes
  - Test forward pass produces correct output shape
  - Test model contains expected layer types
  - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 3. Implement image transformation pipeline
  - Create `src/transforms.py` with `ImageTransformer` class
  - Implement resize transformation to target dimensions
  - Implement normalization with configurable mean and std
  - Implement PIL/NumPy to tensor conversion
  - Implement inference transformation pipeline
  - Implement training transformation pipeline with augmentations (random flips, rotations, color jitter)
  - _Requirements: 3.1, 3.2, 3.3, 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ]* 3.1 Write property test for resize transformation
  - **Property 6: Resize transformation preserves dimensions**
  - **Validates: Requirements 3.1, 7.1**

- [ ]* 3.2 Write property test for normalization
  - **Property 7: Normalization produces expected statistics**
  - **Validates: Requirements 3.2, 7.2**

- [ ]* 3.3 Write property test for tensor conversion
  - **Property 8: Tensor conversion produces valid format**
  - **Validates: Requirements 3.3, 7.3**

- [ ]* 3.4 Write property test for augmentation
  - **Property 9: Augmentation produces varied outputs**
  - **Validates: Requirements 7.4**

- [ ]* 3.5 Write property test for pipeline consistency
  - **Property 10: Transformation pipeline consistency**
  - **Validates: Requirements 7.5**

- [ ]* 3.6 Write unit tests for transformation edge cases
  - Test with very small images (10x10)
  - Test with very large images (4000x4000)
  - Test with grayscale images converted to RGB
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 4. Implement inference engine
  - Create `src/predictor.py` with `DiseasePredictor` class
  - Implement model loading from file path
  - Implement device selection (CPU/GPU)
  - Implement predict method that takes tensor and returns prediction
  - Implement top-k predictions method
  - Create `Prediction` dataclass with disease_name, confidence, class_index
  - Load class names from configuration file
  - _Requirements: 4.1, 4.2, 4.3_

- [ ]* 4.1 Write property test for model processing
  - **Property 11: Model processes valid inputs**
  - **Validates: Requirements 4.1**

- [ ]* 4.2 Write property test for probability distribution
  - **Property 12: Output is valid probability distribution**
  - **Validates: Requirements 4.2**

- [ ]* 4.3 Write property test for prediction logic
  - **Property 13: Prediction matches maximum probability**
  - **Validates: Requirements 4.3**

- [ ]* 4.4 Write unit tests for inference engine
  - Test model loading with valid model file
  - Test prediction with sample tensor
  - Test top-k predictions returns correct number of results
  - Test error handling for missing model file
  - _Requirements: 4.1, 9.3_

- [ ] 5. Implement authentication module
  - Create `src/auth.py` with `AuthenticationManager` class
  - Implement user credential storage (JSON file or SQLite)
  - Implement password hashing with bcrypt
  - Implement authenticate method
  - Implement session creation and validation using Streamlit session state
  - Implement logout method
  - Create `User` and `Session` dataclasses
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ]* 5.1 Write property test for valid credentials
  - **Property 1: Valid credentials grant access**
  - **Validates: Requirements 1.1**

- [ ]* 5.2 Write property test for invalid credentials
  - **Property 2: Invalid credentials are rejected**
  - **Validates: Requirements 1.2**

- [ ]* 5.3 Write property test for session lifecycle
  - **Property 3: Session lifecycle integrity**
  - **Validates: Requirements 1.3, 1.4**

- [ ]* 5.4 Write unit tests for authentication
  - Test session creation after successful login
  - Test session validation
  - Test logout clears session
  - _Requirements: 1.3, 1.4_

- [ ] 6. Implement file upload and validation
  - Create `src/upload.py` with file validation functions
  - Implement supported format validation (JPEG, PNG, JPG)
  - Implement file size validation
  - Implement image loading and validation
  - Create `UploadedImage` dataclass
  - _Requirements: 2.2, 2.3_

- [ ]* 6.1 Write property test for format validation
  - **Property 4: Supported image formats are accepted**
  - **Validates: Requirements 2.2**

- [ ]* 6.2 Write unit tests for upload validation
  - Test rejection of unsupported formats (PDF, TXT, etc.)
  - Test handling of corrupted image files
  - Test file size limits
  - _Requirements: 2.3_

- [ ] 7. Implement error handling utilities
  - Create `src/errors.py` with custom exception classes
  - Implement error message formatting functions
  - Implement error logging setup
  - Create user-friendly error message mappings
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ]* 7.1 Write property test for error handling
  - **Property 17: Errors produce user-friendly messages**
  - **Validates: Requirements 9.1, 9.2, 9.4**

- [ ] 8. Build Streamlit login page
  - Create login UI with username and password inputs
  - Integrate with AuthenticationManager
  - Display error messages for failed login
  - Redirect to upload page on successful login
  - Store authentication state in session
  - _Requirements: 1.1, 1.2_

- [ ] 9. Build Streamlit upload page
  - Create file uploader component
  - Display upload instructions
  - Validate uploaded files using upload module
  - Display image preview after successful upload
  - Add error handling for invalid uploads
  - Require authentication to access page
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ]* 9.1 Write property test for upload preview
  - **Property 5: Upload success displays preview**
  - **Validates: Requirements 2.4**

- [ ] 10. Build prediction and results display
  - Create analyze button that triggers prediction
  - Transform uploaded image using ImageTransformer
  - Run inference using DiseasePredictor
  - Display predicted disease name
  - Display confidence score as percentage
  - Display warning for low confidence predictions (< 60%)
  - Create `DiagnosisResult` dataclass
  - _Requirements: 4.1, 5.1, 5.2, 5.4_

- [ ]* 10.1 Write property test for disease name display
  - **Property 14: Results contain disease name**
  - **Validates: Requirements 5.1**

- [ ]* 10.2 Write property test for confidence formatting
  - **Property 15: Confidence displayed as percentage**
  - **Validates: Requirements 5.2**

- [ ]* 10.3 Write property test for low confidence warning
  - **Property 16: Low confidence triggers warning**
  - **Validates: Requirements 5.4**

- [ ] 11. Implement navigation and UI flow
  - Add sidebar with navigation options
  - Add logout button in sidebar
  - Implement page routing based on authentication state
  - Ensure logical flow: login → upload → results
  - Add "Upload Another" button on results page
  - _Requirements: 8.4_

- [ ] 12. Integrate all components in main application
  - Create `app.py` that initializes all components
  - Load model at startup
  - Set up error logging
  - Configure Streamlit page settings
  - Wire together authentication, upload, transformation, and prediction
  - Add error handling at application level
  - _Requirements: 4.1, 9.2, 9.3_

- [ ]* 12.1 Write integration tests for complete pipeline
  - Test end-to-end flow: login → upload → predict → display
  - Test with sample crop disease images
  - Test error scenarios: corrupted images, missing model
  - _Requirements: 1.1, 2.2, 3.1, 4.1, 5.1_

- [x] 13. Create sample model and test data
  - Create a simple trained model or random initialized model for testing
  - Save model weights to `models/` directory
  - Create `config/class_names.json` with sample disease classes
  - Create `config/model_config.json` with normalization stats and model parameters
  - Add sample test images to `data/test_images/`
  - _Requirements: 4.1, 6.1_

- [ ] 14. Add documentation and setup instructions
  - Create README.md with project overview
  - Document installation steps
  - Document how to run the application
  - Document how to run tests
  - Add code comments for complex functions
  - Document model architecture and training requirements
  - _Requirements: All_

- [ ] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
