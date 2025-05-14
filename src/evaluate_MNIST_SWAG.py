import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils.datasets import load_MNIST, get_ood_mnist
from utils.models import LeNet, SWAG
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
    _, _, test_loader = load_MNIST(batch_size=args.batch_size)
    
    # Create and load model
    base_model = LeNet().to(device)
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
    
    # Print results
    print('\nMNIST Test Set Results:')
    print(f'Accuracy: {test_accuracy:.2f}%')
    print(f'Average Confidence: {mean_confidence:.4f}')
    print(f'Expected Calibration Error: {ece:.4f}')
    print(f'Average Loss: {test_loss:.4f}')
    print(f'AUROC (Confidence vs Correctness): {auroc:.4f}')
    print(f'Mean Confidence (Correct Predictions): {mean_confidence_correct:.4f}')
    print(f'Mean Confidence (Incorrect Predictions): {mean_confidence_incorrect:.4f}')
    
    # Optional: OOD evaluation
    if args.ood_dataset:
        ood_loader = get_ood_mnist(args.ood_dataset, batch_size=args.batch_size)
        print(f'\nEvaluating OOD detection on {args.ood_dataset.upper()}...')
        ood_confidences = []
        
        with torch.no_grad():
            for data, _ in ood_loader:
                data = data.to(device)
                
                # Monte Carlo sampling for OOD data
                predictions = []
                for _ in range(args.num_samples):
                    swag_model.sample()
                    adjust_bn(swag_model, test_loader, device)
                    swag_model.eval()
                    output = swag_model(data)
                    predictions.append(F.softmax(output, dim=1))
                
                # Average predictions
                predictions = torch.stack(predictions)
                mean_prediction = predictions.mean(0)
                
                # Store confidences
                confidence = mean_prediction.max(1)[0]
                ood_confidences.extend(confidence.cpu().numpy())
        
        # Calculate OOD AUROC
        ood_confidences = np.array(ood_confidences)
        labels = np.concatenate([np.ones_like(all_confidences), np.zeros_like(ood_confidences)])
        scores = np.concatenate([all_confidences, ood_confidences])
        ood_auroc = roc_auc_score(labels, scores)
        
        print('\nOOD Detection Results:')
        print(f'OOD AUROC: {ood_auroc:.4f}')
        print(f'In-distribution mean confidence: {np.mean(all_confidences):.4f}')
        print(f'OOD mean confidence: {np.mean(ood_confidences):.4f}')
    
    print(f'\nEvaluation time: {time.time() - start_time:.2f}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SWAG evaluation for MNIST')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for evaluation (default: 128)')
    parser.add_argument('--num-samples', type=int, default=30,
                        help='number of Monte Carlo samples (default: 30)')
    parser.add_argument('--max-models', type=int, default=40,
                        help='maximum number of SWAG models (default: 40)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='path to saved SWAG model')
    parser.add_argument('--ood-dataset', type=str, choices=['emnist', 'fmnist', 'kmnist'],
                        help='OOD dataset to evaluate on (optional)')
    
    args = parser.parse_args()
    evaluate_swag(args) 