import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

def evaluate(model, loader, ood=False):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_probs, all_preds, all_labels, all_imgs = [], [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y)
            all_imgs.append(x.cpu())
            correct += (preds == y).sum().item()
            total += y.size(0)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    imgs = torch.cat(all_imgs)

    acc = correct / total
    conf = probs.max(dim=1).values.mean().item()
    scores = probs.max(dim=1).values.cpu().numpy()
    targets = [0] * len(scores) if ood else [1] * len(scores)
    return acc, conf, scores, targets, preds, labels, imgs


def fpr95(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    try:
        return fpr[np.where(tpr >= 0.95)[0][0]]
    except IndexError:
        return 1.0


def evaluate_la(la_model, loader, ood=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_probs, all_preds, all_labels, all_imgs = [], [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = la_model(x, pred_type='glm', link_approx='probit')
            preds = probs.argmax(dim=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y)
            all_imgs.append(x.cpu())

            correct += (preds==y).sum().item()
            total += y.size(0)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    imgs = torch.cat(all_imgs)

    acc = correct / total
    conf = probs.max(dim=1).values.mean().item()
    scores = probs.max(dim=1).values.cpu().numpy()
    targets = [0] * len(scores) if ood else [1] * len(scores)

    return acc, conf, scores, targets, preds, labels, imgs


def visualize(images, preds, labels, title="Images"):
    plt.figure(figsize=(12, 4))
    for i in range(min(10, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Pred:{preds[i]}, Label:{labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def accuracy(outputs, targets):
    """Calculate classification accuracy.
    
    Args:
        outputs: Model outputs (logits or probabilities)
        targets: Ground truth labels
        
    Returns:
        float: Classification accuracy
    """
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return correct / total

def calibration_error(predictions, targets, confidences, n_bins=10):
    """Calculate Expected Calibration Error (ECE).
    
    Args:
        predictions: Predicted classes
        targets: True classes
        confidences: Prediction confidences
        n_bins: Number of bins for confidence scores
        
    Returns:
        float: Expected Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = predictions == targets
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in current bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece