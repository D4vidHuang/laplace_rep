import os
import torch
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

from utils.datasets import get_mnist, get_ood_mnist
from utils.models import LeNet, MLP


def evaluate_de(models, loader, ood=False):
    """
    similar to metrics.evaluate
    use a list of models for softmax
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for m in models:
        m.eval()
        m.to(device)

    all_probs, all_preds, all_labels = [], [], []
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # stack softmax
            probs_stack = torch.stack([m(x).softmax(1) for m in models], dim=0)
            probs = probs_stack.mean(0)
            preds = probs.argmax(1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(y)

            correct += (preds == y).sum().item()
            total += y.size(0)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    acc = correct / total
    conf = probs.max(1).values.mean().item()
    scores = probs.max(1).values.cpu().numpy()
    # targets: ID→1, OOD→0
    targets = [0] * len(scores) if ood else [1] * len(scores)

    return acc, conf, scores, targets, preds, labels


def run_evaluation(model_name, ood_dataset, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. load 5 MAP
    model_cls = MLP if model_name == 'mlp' else LeNet
    model_paths = [os.path.join('models', f'MNIST_de_model_{i + 1}.pt') for i in range(5)]
    models = []
    for p in model_paths:
        m = model_cls().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        models.append(m)

    # 2. ID
    _, test_loader = get_mnist(batch_size=batch_size)
    acc_id, conf_id, scores_id, targets_id, preds_id, labels_id = evaluate_de(models, test_loader, ood=False)

    # ID AUROC
    binary_targets = (preds_id == labels_id).cpu().numpy().astype(int)
    binary_scores = scores_id
    cls_auroc = roc_auc_score(binary_targets, binary_scores)

    # 3. OOD
    ood_loader = get_ood_mnist(ood_dataset, batch_size=batch_size)
    acc_ood, conf_ood, scores_ood, targets_ood, *_ = evaluate_de(models, ood_loader, ood=True)

    all_scores = np.concatenate([scores_id, scores_ood])
    all_targets = np.concatenate([targets_id, targets_ood])
    ood_auroc = roc_auc_score(all_targets, all_scores)

    return acc_id, conf_id, cls_auroc, acc_ood, conf_ood, ood_auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'lenet'], default='lenet')
    parser.add_argument('--ood', choices=['emnist', 'fmnist', 'kmnist'], default='emnist')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    acc, conf, cls_auroc, ood_acc, ood_conf, ood_auroc = \
        run_evaluation(args.model, args.ood, args.batch_size)

    print(f"[DE] ID Accuracy: {acc * 100:.2f}% | Confidence: {conf:.4f} | "
          f"AUROC: {cls_auroc:.4f}")
    print(f"[DE] OOD={args.ood.upper()} Accuracy: {ood_acc * 100:.2f}% | "
          f"Confidence: {ood_conf*100:.3f} | AUROC: {ood_auroc*100:.3f}")
