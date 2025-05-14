import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.datasets import get_german_credit
from utils.models import GermanCreditMLP
from laplace import Laplace
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

def evaluate(model, loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_probs, all_preds, all_labels = [], [], []
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y)
            correct += (preds == y).sum().item()
            total += y.size(0)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    acc = correct / total
    conf = probs.max(dim=1).values.mean().item()
    uncertainty = 1 - probs.max(dim=1).values
    return acc, conf, probs, preds, labels, uncertainty

def evaluate_la(la_model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_probs, all_preds, all_labels = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = la_model(x, pred_type='glm', link_approx='probit')
            preds = probs.argmax(dim=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y)

            correct += (preds==y).sum().item()
            total += y.size(0)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    acc = correct / total
    conf = probs.max(dim=1).values.mean().item()
    uncertainty = 1 - probs.max(dim=1).values
    return acc, conf, probs, preds, labels, uncertainty

def plot_roc_curve(y_true, y_scores, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc

def plot_uncertainty_histogram(uncertainty, labels, title='Uncertainty Distribution'):
    plt.figure(figsize=(10, 6))

    correct_indices = labels == 0
    incorrect_indices = labels == 1
    
    plt.hist(uncertainty[correct_indices].cpu().numpy(), alpha=0.5, bins=30, label='Good Credit')
    plt.hist(uncertainty[incorrect_indices].cpu().numpy(), alpha=0.5, bins=30, label='Bad Credit')
    
    plt.xlabel('Uncertainty (1 - max probability)')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_calibration_curve(y_true, y_probs, n_bins=10, title='Calibration Curve'):

    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.plot(prob_pred, prob_true, "s-", label=f"{title}")
    
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    

    ece = expected_calibration_error(y_true, y_probs, n_bins)
    return ece

def expected_calibration_error(y_true, y_prob, n_bins=10):

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = np.zeros(n_bins)
    confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):

        in_bin = np.logical_and(y_prob >= bin_lower, y_prob < bin_upper)
        bin_counts[i] = np.sum(in_bin)
        
        if bin_counts[i] > 0:

            accuracies[i] = np.mean(y_true[in_bin] == np.round(y_prob[in_bin]))
            confidences[i] = np.mean(y_prob[in_bin])
    

    ece = np.sum(bin_counts * np.abs(accuracies - confidences)) / np.sum(bin_counts)
    return ece

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden layer size')
    parser.add_argument('--mode', choices=['map', 'la', 'la_star'], default='map',
                       help='model type: map=point estimate, la=Laplace with Kron, la_star=Laplace with Full')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    _, test_loader = get_german_credit(batch_size=args.batch_size)

    for x, _ in test_loader:
        input_size = x.size(1)
        print(f"Feature dimension: {input_size}")
        break

    model = GermanCreditMLP(input_size=input_size, hidden_size=args.hidden_size)
    
    if args.mode == 'map':
        model.load_state_dict(torch.load('models/german_credit_map.pt', map_location=device))
        model.to(device)
        eval_func = evaluate
        eval_model = model
    elif args.mode in ['la', 'la_star']:
        model.load_state_dict(torch.load('models/german_credit_map.pt', map_location=device))
        hessian = 'kron' if args.mode == 'la' else 'full'
        la = Laplace(model,
                     likelihood='classification',
                     subset_of_weights='last_layer',
                     hessian_structure=hessian)
        la.load_state_dict(torch.load(f'models/german_credit_{args.mode}.pt', map_location=device))
        eval_func = evaluate_la
        eval_model = la
    
    acc, conf, probs, preds, labels, uncertainty = eval_func(eval_model, test_loader)
    

    print(f"[{args.mode.upper()}] Test Accuracy: {acc * 100:.2f}%, Confidence: {conf:.4f}")
    

    cm = confusion_matrix(labels.cpu(), preds.cpu())
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels.cpu(), preds.cpu(), target_names=['Good Credit', 'Bad Credit']))
    

    if probs.shape[1] == 2: 
        scores = probs[:, 1].cpu().numpy()
        true_labels = labels.cpu().numpy()
        auroc = plot_roc_curve(true_labels, scores, title=f"{args.mode.upper()} ROC Curve")
        print(f"[{args.mode.upper()}] AUROC: {auroc:.4f}")
        
        ece = plot_calibration_curve(true_labels, scores, title=f"{args.mode.upper()} Calibration Curve")
        print(f"[{args.mode.upper()}] Expected Calibration Error: {ece:.4f}")
    
    plot_uncertainty_histogram(uncertainty, labels, title=f"{args.mode.upper()} Uncertainty Distribution") 