import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils.datasets import get_cifar10
from utils.models import CIFAR10CNN, SWAG
from utils.metrics import accuracy, calibration_error
from sklearn.metrics import roc_auc_score
import time

def adjust_bn(model, loader, device):
    """
    Adjust batch normalization statistics using the entire loader
    """
    model.train()
    for images, _ in loader:
        images = images.to(device)
        model(images)

def evaluate_swag(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    _, test_loader = get_cifar10(batch_size=args.batch_size)
    
    # Create and load model
    base_model = CIFAR10CNN().to(device)
    swag_model = SWAG(base_model, max_models=args.max_models).to(device)
    swag_model.load_state_dict(torch.load(args.model_path))
    
    # Evaluation loop
    print("\nStarting SWAG evaluation...")
    all_predictions = []
    all_targets = []
    all_confidences = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Monte Carlo sampling
            predictions = []
            for _ in range(args.num_samples):
                # Sample new weights
                swag_model.sample()
                
                # Update batch normalization statistics
                adjust_bn(swag_model, test_loader, device)
                
                # Make prediction
                swag_model.eval()
                output = swag_model(data)
                predictions.append(F.softmax(output, dim=1))
            
            # Average predictions
            predictions = torch.stack(predictions)
            mean_prediction = predictions.mean(0)
            
            # Calculate metrics
            confidence, predicted = mean_prediction.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Store predictions and targets for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_probs.append(mean_prediction.cpu().numpy())
            
            # Calculate loss
            test_loss += criterion(mean_prediction.log(), target).item()
    
    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    
    # Calculate metrics
    test_accuracy = 100. * correct / total
    test_loss /= len(test_loader)
    mean_confidence = np.mean(all_confidences)
    ece = calibration_error(all_predictions, all_targets, all_confidences)
    
    # Calculate AUROC using correct/incorrect predictions
    correct_predictions = (all_predictions == all_targets)
    auroc = roc_auc_score(correct_predictions, all_confidences)
    
    # Calculate mean confidence for correct and incorrect predictions
    mean_confidence_correct = np.mean(all_confidences[correct_predictions])
    mean_confidence_incorrect = np.mean(all_confidences[~correct_predictions])
    
    # Calculate per-class accuracy
    per_class_acc = []
    for c in range(10):
        mask = (all_targets == c)
        if np.sum(mask) > 0:  # avoid division by zero
            class_acc = np.mean(all_predictions[mask] == c) * 100
            per_class_acc.append(class_acc)
            print(f'Class {c} Accuracy: {class_acc:.2f}%')
    
    # Print results
    print('\nCIFAR10 Test Set Results:')
    print(f'Accuracy: {test_accuracy:.2f}%')
    print(f'Average Confidence: {mean_confidence:.4f}')
    print(f'Expected Calibration Error: {ece:.4f}')
    print(f'Average Loss: {test_loss:.4f}')
    print(f'AUROC (Confidence vs Correctness): {auroc:.4f}')
    print(f'Mean Confidence (Correct Predictions): {mean_confidence_correct:.4f}')
    print(f'Mean Confidence (Incorrect Predictions): {mean_confidence_incorrect:.4f}')
    
    # Calculate predictive entropy
    all_probs = np.concatenate(all_probs, axis=0)
    entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)
    mean_entropy = np.mean(entropy)
    mean_entropy_correct = np.mean(entropy[correct_predictions])
    mean_entropy_incorrect = np.mean(entropy[~correct_predictions])
    
    print('\nPredictive Uncertainty:')
    print(f'Mean Predictive Entropy: {mean_entropy:.4f}')
    print(f'Mean Entropy (Correct Predictions): {mean_entropy_correct:.4f}')
    print(f'Mean Entropy (Incorrect Predictions): {mean_entropy_incorrect:.4f}')
    
    print(f'\nEvaluation time: {time.time() - start_time:.2f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SWAG evaluation for CIFAR10')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for evaluation (default: 128)')
    parser.add_argument('--num-samples', type=int, default=30,
                        help='number of Monte Carlo samples (default: 30)')
    parser.add_argument('--max-models', type=int, default=40,
                        help='maximum number of SWAG models (default: 40)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to saved SWAG model')
    
    args = parser.parse_args()
    evaluate_swag(args) 