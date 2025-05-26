import os
import torch
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

from utils.datasets import get_mnist, get_ood_mnist
from utils.models import LeNet, MLP


def evaluate_de(models, loader, ood=False):
    """
    和 metrics.evaluate 相似，
    但预测时用模型列表做 softmax 平均。
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
            # stack 每个模型的 softmax 概率
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
    # targets: ID→1, OOD→0，与 metrics.evaluate 保持一致
    targets = [0] * len(scores) if ood else [1] * len(scores)

    return acc, conf, scores, targets, preds, labels


def run_evaluation(model_name, ood_dataset, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载 5 个独立训练好的 MAP 模型
    model_cls = MLP if model_name == 'mlp' else LeNet
    model_paths = [os.path.join('models', f'MNIST_de_model_{i + 1}.pt') for i in range(5)]
    models = []
    for p in model_paths:
        m = model_cls().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        models.append(m)

    # 2) ID 上评估
    _, test_loader = get_mnist(batch_size=batch_size)
    acc_id, conf_id, scores_id, targets_id, preds_id, labels_id = evaluate_de(models, test_loader, ood=False)

    # 分类准确率 vs 错误 的 AUROC（可选，可印证 ensemble 效果）
    binary_targets = (preds_id == labels_id).cpu().numpy().astype(int)
    binary_scores = scores_id
    cls_auroc = roc_auc_score(binary_targets, binary_scores)

    # 3) OOD 上评估
    ood_loader = get_ood_mnist(ood_dataset, batch_size=batch_size)
    acc_ood, conf_ood, scores_ood, targets_ood, *_ = evaluate_de(models, ood_loader, ood=True)

    # OOD 检测 AUROC：ID (1) vs OOD (0)
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

    print(f"[DE] ID Accuracy: {acc * 100:.2f}%, Confidence: {conf:.4f}, "
          f"AUROC: {cls_auroc:.4f}")
    print(f"[DE] OOD={args.ood.upper()} Accuracy: {ood_acc * 100:.2f}%, "
          f"Confidence: {ood_conf:.4f}, AUROC: {ood_auroc:.4f}")
