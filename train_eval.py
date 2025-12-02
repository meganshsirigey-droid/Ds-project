"""
Training and evaluation utilities
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def train_one_epoch(model, train_loader, device, optimizer, criterion=None):
    """
    Train the model for one epoch.
    Optimized for speed and memory efficiency.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        device: torch.device (cuda or cpu)
        optimizer: Optimizer
        criterion: Loss function (default: CrossEntropyLoss)
    
    Returns:
        Average loss and accuracy for the epoch
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use non_blocking for faster GPU transfer
    non_blocking = device.type == 'cuda'
    
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Compute accuracy efficiently
        running_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        batch_total = labels.size(0)
        batch_correct = (predicted == labels).sum().item()
        total += batch_total
        correct += batch_correct
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, device, criterion=None):
    """
    Evaluate the model on test data.
    Optimized for speed and memory efficiency.
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: torch.device (cuda or cpu)
        criterion: Loss function (default: CrossEntropyLoss)
    
    Returns:
        Average loss and accuracy
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use non_blocking for faster GPU transfer
    non_blocking = device.type == 'cuda'
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Compute accuracy efficiently
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            batch_total = labels.size(0)
            batch_correct = (predicted == labels).sum().item()
            total += batch_total
            correct += batch_correct
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def save_model(model, filepath, optimizer=None, epoch=None, accuracy=None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        accuracy: Optional accuracy value
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath, device):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model instance
        filepath: Path to the saved model
        device: torch.device (cuda or cpu)
    
    Returns:
        Dictionary with loaded state (epoch, accuracy, etc.)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    result = {}
    if 'epoch' in checkpoint:
        result['epoch'] = checkpoint['epoch']
    if 'accuracy' in checkpoint:
        result['accuracy'] = checkpoint['accuracy']
    
    print(f"Model loaded from {filepath}")
    return result

