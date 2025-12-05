"""
Training script for the Crop Disease Classifier.

This script trains the CNN model on the rice leaf diseases dataset with:
- Train/validation split
- Data augmentation
- Learning rate scheduling
- Model checkpointing
- Training metrics visualization
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import CropDiseaseClassifier
from src.transforms import ImageTransformer


class RiceLeafDataset(Dataset):
    """Dataset class for rice leaf disease images."""
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with disease class subdirectories
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Get class names from subdirectories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
            for img_path in class_dir.glob('*.JPG'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_image_transformer(config):
    """Get ImageTransformer instance from config."""
    
    input_size = tuple(config['model']['input_size'])
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    
    return ImageTransformer(
        target_size=input_size,
        mean=mean,
        std=std
    )


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def plot_training_history(history, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to: {save_path}")


def train_model():
    """Main training function."""
    
    # Load configuration
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'config' / 'model_config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 60)
    print("Crop Disease Classifier - Training")
    print("=" * 60)
    
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_path = base_dir / 'data' / 'rice_leaf_diseases'
    
    # Create ImageTransformer
    image_transformer = get_image_transformer(config)
    
    # Get transforms from ImageTransformer
    train_transform = image_transformer.get_training_transforms()
    val_transform = image_transformer.get_inference_transforms()
    
    # Create full dataset
    full_dataset = RiceLeafDataset(dataset_path, transform=train_transform)
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    
    # Split into train and validation
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    num_classes = len(full_dataset.classes)
    model = CropDiseaseClassifier(num_classes=num_classes)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = base_dir / 'models' / 'best_model.pth'
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Total training time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_model_path = base_dir / config['model']['model_path']
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    
    # Update class names in config
    class_names_path = base_dir / 'config' / 'class_names.json'
    class_names_data = {'classes': full_dataset.classes}
    with open(class_names_path, 'w') as f:
        json.dump(class_names_data, f, indent=2)
    print(f"Class names updated in: {class_names_path}")
    
    # Update model config with actual number of classes
    config['model']['num_classes'] = num_classes
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model config updated with {num_classes} classes")
    
    # Plot training history
    plot_path = base_dir / 'models' / 'training_history.png'
    plot_training_history(history, plot_path)
    
    # Save training history as JSON
    history_path = base_dir / 'models' / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    print("\n✓ Training complete! You can now use the trained model for inference.")


if __name__ == "__main__":
    train_model()
