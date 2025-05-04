import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model(model, loader, is_ood=False, is_laplace=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not is_laplace:
        model.eval()

    all_probs, all_preds, all_labels, all_imgs = [], [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            if is_laplace:
                probs = model(x, pred_type='glm', link_approx='probit')
            else:
                logits = model(x)
                probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_imgs.append(x.cpu())
            correct += (preds == y).sum().item()
            total += y.size(0)

    probs = torch.cat(all_probs)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    imgs = torch.cat(all_imgs)

    acc = correct / total
    conf = probs.max(dim=1).values.mean().item()
    scores = probs.max(dim=1).values.numpy()
    targets = [0] * len(scores) if is_ood else [1] * len(scores)

    return acc, conf, scores, targets, preds, labels, imgs

def compute_ood_metrics(id_scores, id_targets, ood_scores, ood_targets):
    all_scores = np.concatenate([id_scores, ood_scores])
    all_targets = np.concatenate([id_targets, ood_targets])
    auroc = roc_auc_score(all_targets, all_scores)

    fpr, tpr, _ = roc_curve(all_targets, all_scores)
    try:
        fpr95 = fpr[np.where(tpr >= 0.95)[0][0]]
    except IndexError:
        fpr95 = 1.0

    return auroc, fpr95

def visualize_samples(images, preds, labels, title, save_path=None):
    plt.figure(figsize=(12, 4))
    for i in range(min(10, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Pred:{preds[i]}, Label:{labels[i]}")
        plt.axis('off')
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()