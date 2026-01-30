"""
Testing and evaluation script for the trained model.
"""

import torch
import numpy as np
from utils import calculate_metrics, print_metrics

def test_model(model, test_loader, device='cpu', num_classes=3):
    """
    Evaluate the model on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: Device to test on
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*50)
    print("TESTING MODEL")
    print("="*50)
    
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, num_classes)
    
    # Print metrics
    class_names = ['negative', 'neutral', 'positive']
    print_metrics(metrics, class_names)
    
    return metrics