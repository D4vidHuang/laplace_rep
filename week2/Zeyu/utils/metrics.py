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
    #la_model.eval()
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