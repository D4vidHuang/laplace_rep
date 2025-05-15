'''Evaluate CIFAR10 HMC models'''
from __future__ import print_function
import sys
import os

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from resources.csgmcmc.models import *
import numpy as np
from glob import glob
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='HMC CIFAR10 Evaluation')
parser.add_argument('--device_id', type=int, help='device id to use')
parser.add_argument('--num_samples', type=int, default=20,
                    help='number of HMC samples to evaluate')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()

# Data and model directories
data_path = os.path.join(project_root, 'data')
model_dir = os.path.join(project_root, 'models')

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                      download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                       shuffle=False, num_workers=0)

# Model
print('==> Building model..')
net = ResNet18()
if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True

criterion = nn.CrossEntropyLoss()

def compute_metrics(predictions, targets):
    # Convert predictions to probabilities
    probs = torch.softmax(predictions, dim=1)
    
    # Get predicted class and confidence
    confidence, predicted = torch.max(probs, dim=1)
    
    # Compute accuracy
    correct = predicted.eq(targets).float()
    accuracy = correct.mean().item()
    
    # Compute mean confidence
    mean_confidence = confidence.mean().item()
    
    # Compute AUROC (one-vs-rest for multiclass)
    probs_np = probs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Compute AUROC for each class
    auroc_scores = []
    for i in range(10):  # CIFAR10 has 10 classes
        binary_targets = (targets_np == i).astype(np.float32)
        if len(np.unique(binary_targets)) > 1:  # Only compute AUROC if both classes present
            auroc = roc_auc_score(binary_targets, probs_np[:, i])
            auroc_scores.append(auroc)
    
    mean_auroc = np.mean(auroc_scores)
    
    return accuracy, mean_confidence, mean_auroc

def test_ensemble():
    net.eval()
    
    # Load all available models
    model_files = sorted(glob(os.path.join(model_dir, 'cifar10_hmc_epoch_*.pt')))
    if len(model_files) == 0:
        print('No models found in directory:', model_dir)
        return
    
    print(f'Found {len(model_files)} models')
    
    # Select the last num_samples models (or all if less than num_samples)
    model_files = model_files[-args.num_samples:]
    print(f'Using {len(model_files)} models for ensemble prediction')
    
    ensemble_predictions = []
    ensemble_accuracies = []
    
    # Collect all targets
    all_targets = []
    with torch.no_grad():
        for inputs, targets in testloader:
            all_targets.append(targets)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Evaluate each model
    for model_file in model_files:
        print(f'\nEvaluating model: {model_file}')
        checkpoint = torch.load(model_file)
        net.load_state_dict(checkpoint['net'])
        
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in testloader:
                if use_cuda:
                    inputs = inputs.cuda(device_id)
                outputs = net(inputs)
                predictions.append(outputs.cpu())
            
        predictions = torch.cat(predictions, dim=0)
        accuracy, confidence, auroc = compute_metrics(predictions, all_targets)
        print(f'Individual model metrics:')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Confidence: {confidence:.4f}')
        print(f'  AUROC: {auroc:.4f}')
        
        ensemble_accuracies.append(accuracy)
        ensemble_predictions.append(predictions)
    
    # Compute ensemble predictions
    ensemble_predictions = torch.stack(ensemble_predictions)
    mean_predictions = torch.mean(ensemble_predictions, dim=0)
    
    # Compute ensemble metrics
    ensemble_accuracy, ensemble_confidence, ensemble_auroc = compute_metrics(mean_predictions, all_targets)
    
    print('\nFinal Results:')
    print(f'Individual Model Performance (mean ± std):')
    print(f'  Accuracy: {np.mean(ensemble_accuracies):.4f} ± {np.std(ensemble_accuracies):.4f}')
    print(f'\nEnsemble Model Performance:')
    print(f'  Accuracy: {ensemble_accuracy:.4f}')
    print(f'  Confidence: {ensemble_confidence:.4f}')
    print(f'  AUROC: {ensemble_auroc:.4f}')
    
    # Save results to file
    results = {
        'accuracy': ensemble_accuracy,
        'confidence': ensemble_confidence,
        'auroc': ensemble_auroc
    }
    
    results_file = os.path.join(project_root, 'results_hmc.txt')
    with open(results_file, 'w') as f:
        for metric, value in results.items():
            f.write(f'{metric}: {value:.4f}\n')
    
    return results

if __name__ == '__main__':
    test_ensemble() 