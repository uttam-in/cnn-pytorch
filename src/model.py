"""
CNN Model for Crop Disease Classification

This module implements a Convolutional Neural Network for classifying
crop diseases from leaf images using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CropDiseaseClassifier(nn.Module):
    """
    Convolutional Neural Network for crop disease classification.
    
    Architecture:
    - 3 Convolutional blocks with increasing depth (32 -> 64 -> 128 channels)
    - Each block contains 2 Conv2d layers, ReLU activations, MaxPool2d, and Dropout
    - 2 Fully connected layers for classification
    - Softmax activation for probability distribution output
    
    Args:
        num_classes (int): Number of disease classes to predict
    """
    
    def __init__(self, num_classes: int):
        super(CropDiseaseClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Convolutional Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Convolutional Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully Connected Layers
        # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes) # output layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output probability distribution of shape (batch_size, num_classes)
        """
        # Convolutional Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Convolutional Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Convolutional Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        # Softmax for probability distribution
        x = F.softmax(x, dim=1)
        
        return x
