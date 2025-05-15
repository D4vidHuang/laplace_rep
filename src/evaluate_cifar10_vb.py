import torch
import torch.nn.functional as F
from train_cifar10_VB import BayesianCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_vb_cifar10(n_samples=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained VB model
    model = BayesianCNN().to(device)
    model.load_state_dict(torch.load('models/cifar10_vb_best.pt'))
    model.eval()
    
    # Data loading
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_dataset = datasets.CIFAR10('data', train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Evaluation with multiple forward passes
    all_confidences = []
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_predictions = []
            
            # Multiple forward passes for uncertainty estimation
            for _ in range(n_samples):
                output = model(data)
                probs = F.softmax(output, dim=1)
                batch_predictions.append(probs)
            
            # Average predictions over multiple passes
            batch_predictions = torch.stack(batch_predictions)  # [n_samples, batch_size, n_classes]
            mean_predictions = batch_predictions.mean(0)  # [batch_size, n_classes]
            
            # Get predicted class and confidence
            pred_confidence, pred_class = mean_predictions.max(1)
            
            # Update accuracy
            correct += pred_class.eq(target).sum().item()
            total += target.size(0)
            
            # Store results
            all_confidences.extend(pred_confidence.cpu().numpy())
            all_predictions.extend(pred_class.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    avg_confidence = np.mean(all_confidences) * 100
    
    # Calculate AUROC
    # Convert to one-hot format for AUROC calculation
    y_true = np.eye(10)[all_targets]
    y_pred = np.eye(10)[all_predictions]
    auroc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
    
    # Print results in the same format as before
    print(f"Accuracy: {accuracy:.1f}")
    print(f"Average Confidence: {avg_confidence:.1f}")
    print(f"AUROC: {auroc:.3f}")
    
    # Save results
    np.savez('models/cifar10_vb_results.npz',
             accuracy=accuracy,
             confidence=all_confidences,
             predictions=all_predictions,
             targets=all_targets,
             auroc=auroc)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of forward passes for uncertainty estimation')
    args = parser.parse_args()
    
    evaluate_vb_cifar10(args.n_samples) 