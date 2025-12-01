# Design Document

## Overview

The Crop Disease Diagnosis System is a web-based application that combines deep learning with an intuitive user interface to help farmers and agricultural experts identify diseases in rice and pulse crops. The system architecture follows a modular design with clear separation between the authentication layer, user interface, image processing pipeline, and the CNN-based disease classification model.

The application uses Streamlit for the frontend, providing a simple yet powerful interface for image upload and result visualization. The backend leverages PyTorch to implement a Convolutional Neural Network trained on crop disease images. The system processes uploaded leaf images through a transformation pipeline before feeding them to the model for inference.

## Architecture

The system follows a layered architecture:

```
┌─────────────────────────────────────────┐
│         Streamlit Interface             │
│  (Authentication, Upload, Display)      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Application Layer                  │
│  (Session Management, Orchestration)    │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼──────────┐
│ Image Pipeline │  │  CNN Model      │
│ (Transform)    │  │  (Inference)    │
└────────────────┘  └─────────────────┘
```

**Key Architectural Principles:**
- **Separation of Concerns**: UI, business logic, and ML model are decoupled
- **Modularity**: Each component can be developed and tested independently
- **Extensibility**: New disease classes or crop types can be added by retraining the model
- **Simplicity**: Streamlit provides rapid development with minimal boilerplate

## Components and Interfaces

### 1. Authentication Module

**Responsibility**: Manage user login and session state

**Interface**:
```python
class AuthenticationManager:
    def authenticate(username: str, password: str) -> bool
    def create_session(username: str) -> Session
    def validate_session(session_id: str) -> bool
    def logout(session_id: str) -> None
```

**Implementation Notes**:
- Uses Streamlit's session state for maintaining authentication
- Passwords should be hashed using bcrypt or similar
- User credentials stored in a simple JSON file or SQLite database for MVP
- Session tokens stored in Streamlit session state

### 2. Image Transformation Pipeline

**Responsibility**: Preprocess uploaded images for model inference

**Interface**:
```python
class ImageTransformer:
    def __init__(self, target_size: Tuple[int, int], 
                 mean: List[float], std: List[float])
    def transform(self, image: PIL.Image) -> torch.Tensor
    def get_training_transforms(self) -> transforms.Compose
    def get_inference_transforms(self) -> transforms.Compose
```

**Transformation Steps**:
1. Resize to model input dimensions (e.g., 224x224)
2. Convert to tensor
3. Normalize using ImageNet statistics or custom dataset statistics
4. Add batch dimension for inference

**Training Augmentations** (optional for future training):
- Random horizontal/vertical flips
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)
- Random crop

### 3. CNN Model

**Responsibility**: Classify crop diseases from preprocessed images

**Architecture**:
```python
class CropDiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int):
        # Convolutional Block 1
        - Conv2d(3, 32, kernel_size=3, padding=1)
        - ReLU
        - Conv2d(32, 32, kernel_size=3, padding=1)
        - ReLU
        - MaxPool2d(2, 2)
        - Dropout(0.25)
        
        # Convolutional Block 2
        - Conv2d(32, 64, kernel_size=3, padding=1)
        - ReLU
        - Conv2d(64, 64, kernel_size=3, padding=1)
        - ReLU
        - MaxPool2d(2, 2)
        - Dropout(0.25)
        
        # Convolutional Block 3
        - Conv2d(64, 128, kernel_size=3, padding=1)
        - ReLU
        - Conv2d(128, 128, kernel_size=3, padding=1)
        - ReLU
        - MaxPool2d(2, 2)
        - Dropout(0.25)
        
        # Fully Connected Layers
        - Flatten
        - Linear(128 * 28 * 28, 512)
        - ReLU
        - Dropout(0.5)
        - Linear(512, num_classes)
        - Softmax (applied during inference)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Model Specifications**:
- Input: 3-channel RGB images (224x224)
- Output: Probability distribution over disease classes
- Framework: PyTorch
- Activation: ReLU for hidden layers, Softmax for output
- Regularization: Dropout to prevent overfitting

### 4. Inference Engine

**Responsibility**: Load model and perform disease prediction

**Interface**:
```python
class DiseasePredictor:
    def __init__(self, model_path: str, class_names: List[str], 
                 device: str = 'cpu')
    def load_model(self) -> None
    def predict(self, image_tensor: torch.Tensor) -> Prediction
    def get_top_k_predictions(self, image_tensor: torch.Tensor, 
                              k: int = 3) -> List[Prediction]

class Prediction:
    disease_name: str
    confidence: float
    class_index: int
```

**Implementation Notes**:
- Model loaded once at application startup
- Inference runs on CPU by default (GPU optional if available)
- Returns top prediction with confidence score
- Supports batch inference for future scalability

### 5. Streamlit UI Components

**Responsibility**: Provide user interface for all interactions

**Pages/Sections**:
1. **Login Page**: Username/password form
2. **Upload Page**: File uploader with image preview
3. **Results Page**: Disease prediction with confidence score
4. **Sidebar**: Navigation and logout button

**UI Flow**:
```
Login → Upload Image → Preview → Analyze → Display Results → Upload Another/Logout
```

## Data Models

### User
```python
@dataclass
class User:
    username: str
    password_hash: str
    created_at: datetime
    last_login: datetime
```

### Session
```python
@dataclass
class Session:
    session_id: str
    username: str
    created_at: datetime
    is_active: bool
```

### UploadedImage
```python
@dataclass
class UploadedImage:
    filename: str
    image_data: PIL.Image
    upload_timestamp: datetime
    file_size: int
```

### DiagnosisResult
```python
@dataclass
class DiagnosisResult:
    disease_name: str
    confidence: float
    image_filename: str
    timestamp: datetime
    all_predictions: List[Prediction]  # Top-k predictions
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Authentication Properties

**Property 1: Valid credentials grant access**
*For any* valid username and password combination, authentication should succeed and grant platform access.
**Validates: Requirements 1.1**

**Property 2: Invalid credentials are rejected**
*For any* invalid credential combination (wrong password, non-existent user, empty fields), authentication should fail and display an error message.
**Validates: Requirements 1.2**

**Property 3: Session lifecycle integrity**
*For any* user, logging in should create a valid session that persists during interaction, and logging out should invalidate that session requiring re-authentication for subsequent access.
**Validates: Requirements 1.3, 1.4**

### Image Upload and Validation Properties

**Property 4: Supported image formats are accepted**
*For any* image file in supported formats (JPEG, PNG, JPG), the system should accept the upload and allow processing.
**Validates: Requirements 2.2**

**Property 5: Upload success displays preview**
*For any* successfully uploaded valid image, the system should display a preview of that image.
**Validates: Requirements 2.4**

### Image Transformation Properties

**Property 6: Resize transformation preserves dimensions**
*For any* input image of arbitrary size, applying the resize transformation should produce an output with exactly the model's expected input dimensions (e.g., 224x224).
**Validates: Requirements 3.1, 7.1**

**Property 7: Normalization produces expected statistics**
*For any* input image, applying normalization should produce pixel values with mean and standard deviation matching the training dataset statistics (within numerical tolerance).
**Validates: Requirements 3.2, 7.2**

**Property 8: Tensor conversion produces valid format**
*For any* valid PIL or NumPy image, the transformation pipeline should produce a PyTorch tensor with correct shape (C, H, W), dtype (float32), and value range.
**Validates: Requirements 3.3, 7.3**

**Property 9: Augmentation produces varied outputs**
*For any* input image, applying training augmentation transforms multiple times should produce different outputs, demonstrating that random transformations are working.
**Validates: Requirements 7.4**

**Property 10: Transformation pipeline consistency**
*For any* image, the base transformations (resize, normalize, to_tensor) should be applied in the same order for both training and inference pipelines.
**Validates: Requirements 7.5**

### Model Inference Properties

**Property 11: Model processes valid inputs**
*For any* valid preprocessed tensor input, the CNN model should execute forward pass without errors and produce output.
**Validates: Requirements 4.1**

**Property 12: Output is valid probability distribution**
*For any* input image, the model output should be a valid probability distribution: all values between 0 and 1, sum equals 1 (within numerical tolerance), and length equals number of disease classes.
**Validates: Requirements 4.2**

**Property 13: Prediction matches maximum probability**
*For any* model output, the predicted disease class index should correspond to the position of the maximum probability value in the output distribution.
**Validates: Requirements 4.3**

### Results Display Properties

**Property 14: Results contain disease name**
*For any* prediction result, the displayed output should contain the predicted disease name as a string.
**Validates: Requirements 5.1**

**Property 15: Confidence displayed as percentage**
*For any* prediction confidence value between 0 and 1, the displayed confidence should be correctly formatted as a percentage (multiplied by 100, with appropriate decimal places).
**Validates: Requirements 5.2**

**Property 16: Low confidence triggers warning**
*For any* prediction with confidence below 60%, the system should display a warning message indicating diagnostic uncertainty.
**Validates: Requirements 5.4**

### Error Handling Properties

**Property 17: Errors produce user-friendly messages**
*For any* error condition (upload failure, inference failure, file system error), the system should catch the exception, display a user-friendly error message, and allow retry without crashing.
**Validates: Requirements 9.1, 9.2, 9.4**


## Error Handling

The system implements comprehensive error handling at multiple layers:

### Input Validation Errors
- **Invalid file format**: Display message listing supported formats (JPEG, PNG, JPG)
- **File too large**: Display message with maximum file size limit
- **Corrupted image**: Display message asking user to upload a different image
- **Empty upload**: Prevent submission and display instruction message

### Model Errors
- **Model file not found**: Display error at startup with instructions to place model file in correct location
- **Model loading failure**: Log detailed error, display generic message to user
- **Inference failure**: Log stack trace, display message asking user to try different image
- **Out of memory**: Catch exception, suggest using smaller image or restarting application

### Authentication Errors
- **Invalid credentials**: Display message indicating username or password is incorrect
- **Session expired**: Redirect to login page with message
- **Database connection failure**: Display message indicating system is temporarily unavailable

### File System Errors
- **Permission denied**: Display message about file access permissions
- **Disk full**: Display message indicating storage issue
- **Network timeout**: Display message with retry option

**Error Logging Strategy**:
- All errors logged to file with timestamp, error type, and stack trace
- User-facing messages are friendly and actionable
- Sensitive information (passwords, internal paths) never displayed to users
- Critical errors trigger alerts for system administrators

## Testing Strategy

The testing strategy employs both unit testing and property-based testing to ensure comprehensive coverage and correctness.

### Unit Testing

Unit tests verify specific examples, edge cases, and component integration:

**Authentication Module**:
- Test successful login with valid credentials
- Test failed login with invalid credentials
- Test session creation and validation
- Test logout functionality

**Image Transformation**:
- Test resize with specific image sizes
- Test normalization with known pixel values
- Test tensor conversion with sample images
- Test edge cases: very small images, very large images, grayscale images

**CNN Model Architecture**:
- Test model instantiation
- Test forward pass with sample input
- Test output shape matches expected dimensions
- Test model can be saved and loaded

**Inference Engine**:
- Test prediction with sample images
- Test top-k predictions functionality
- Test model loading from file
- Test error handling for invalid inputs

**Streamlit UI** (integration tests):
- Test login flow
- Test image upload flow
- Test results display
- Test navigation between pages

### Property-Based Testing

Property-based tests verify universal properties across many randomly generated inputs. We will use **Hypothesis** (for Python) as the property-based testing library.

**Configuration**: Each property-based test will run a minimum of 100 iterations to ensure thorough coverage of the input space.

**Test Tagging**: Each property-based test will include a comment explicitly referencing the correctness property from this design document using the format: `# Feature: crop-disease-diagnosis, Property X: [property text]`

**Property Test Coverage**:

1. **Authentication properties** (Properties 1-3): Generate random valid/invalid credentials, test authentication behavior and session lifecycle
2. **Image validation properties** (Properties 4-5): Generate files of various formats, test acceptance/rejection and preview display
3. **Transformation properties** (Properties 6-10): Generate random images of various sizes and formats, test transformation invariants
4. **Model inference properties** (Properties 11-13): Generate random valid tensors, test model execution and output validity
5. **Display properties** (Properties 14-16): Generate random prediction results, test display formatting and warning triggers
6. **Error handling properties** (Property 17): Generate various error conditions, test graceful handling and recovery

**Test Data Generators**:
- Random image generator: creates PIL images with random dimensions, pixel values, and formats
- Random credential generator: creates valid and invalid username/password combinations
- Random tensor generator: creates PyTorch tensors with various shapes and value ranges
- Random prediction generator: creates prediction results with various confidence levels

**Property Test Examples**:

```python
# Feature: crop-disease-diagnosis, Property 6: Resize transformation preserves dimensions
@given(st.integers(min_value=50, max_value=2000), 
       st.integers(min_value=50, max_value=2000))
@settings(max_examples=100)
def test_resize_preserves_dimensions(width, height):
    image = generate_random_image(width, height)
    transformer = ImageTransformer(target_size=(224, 224))
    result = transformer.transform(image)
    assert result.shape == (3, 224, 224)

# Feature: crop-disease-diagnosis, Property 12: Output is valid probability distribution
@given(st.integers(min_value=1, max_value=50))
@settings(max_examples=100)
def test_output_is_valid_probability_distribution(num_classes):
    model = CropDiseaseClassifier(num_classes=num_classes)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert torch.all(output >= 0) and torch.all(output <= 1)
    assert torch.isclose(output.sum(), torch.tensor(1.0), atol=1e-6)
    assert output.shape[1] == num_classes
```

### Integration Testing

Integration tests verify the complete pipeline:
- End-to-end test: login → upload → transform → predict → display
- Test with real crop disease images from validation dataset
- Test with edge cases: corrupted images, unsupported formats, very large files
- Test concurrent user sessions

### Test Coverage Goals

- Minimum 80% code coverage for core functionality
- 100% coverage for critical paths (authentication, prediction pipeline)
- All 17 correctness properties must have corresponding property-based tests
- All error handling paths must be tested

### Testing Tools

- **pytest**: Unit test framework
- **Hypothesis**: Property-based testing library
- **pytest-cov**: Coverage reporting
- **Streamlit testing**: For UI component testing
- **torch.testing**: For tensor comparison utilities

## Deployment Considerations

### Model Artifacts
- Trained model weights stored as `.pth` file
- Class names stored in JSON file
- Normalization statistics (mean, std) stored in configuration file

### Configuration
- Environment variables for model paths, database location, secret keys
- Configuration file for model hyperparameters and UI settings

### Dependencies
- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.25+
- Pillow for image processing
- NumPy for numerical operations

### Performance Optimization
- Model loaded once at startup (not per request)
- Image preprocessing cached where possible
- Consider model quantization for faster inference
- Optional GPU support for faster predictions

### Security
- Passwords hashed with bcrypt
- Session tokens use secure random generation
- File uploads validated and sanitized
- Rate limiting on login attempts to prevent brute force

### Scalability
- Current design supports single-user deployment
- For multi-user: add proper database (PostgreSQL), session management (Redis)
- For high traffic: add load balancer, model serving infrastructure (TorchServe)
- Consider cloud deployment (AWS, GCP, Azure) for production use
