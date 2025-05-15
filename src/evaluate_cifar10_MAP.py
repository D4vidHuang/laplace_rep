import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.datasets import get_cifar10
from utils.models import CIFAR10CNN
from laplace import Laplace
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report

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

def run_evaluation(mode='map', batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = CIFAR10CNN()
    
    # 根据模式加载模型
    if mode == 'map':
        model.load_state_dict(torch.load('models/CIFAR10_map.pt', map_location=device))
        model.to(device)
        eval_func = evaluate
        eval_model = model
    elif mode in ['la', 'la_star']:
        model.load_state_dict(torch.load('models/CIFAR10_map.pt', map_location=device))
        hessian = 'kron' if mode == 'la' else 'full'
        la = Laplace(model,
                     likelihood='classification',
                     subset_of_weights='last_layer',
                     hessian_structure=hessian)
        la.load_state_dict(torch.load(f'models/CIFAR10_{mode}.pt', map_location=device))
        eval_func = evaluate_la
        eval_model = la
    
    # 加载测试数据
    _, test_loader = get_cifar10(batch_size=batch_size)
    
    # 评估模型
    acc, conf, probs, preds, labels, uncertainty = eval_func(eval_model, test_loader)
    
    # 创建二分类问题来计算AUROC
    # 将预测正确的样本标记为正类，预测错误的样本标记为负类
    binary_targets = (preds == labels).cpu().numpy().astype(int)
    binary_scores = uncertainty.cpu().numpy()  # 使用不确定性作为预测分数
    
    # 计算AUROC（注意：这里我们使用不确定性作为分数，所以较大的值应该对应错误的预测）
    auroc = roc_auc_score(1 - binary_targets, binary_scores)
    
    return acc, conf, auroc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['map', 'la', 'la_star'], default='map',
                       help='model type: map=point estimate, la=Laplace with Kron, la_star=Laplace with Full')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for evaluation')
    args = parser.parse_args()

    # 运行评估
    acc, conf, auroc = run_evaluation(args.mode, args.batch_size)
    
    # 打印结果
    print(f"[{args.mode.upper()}] Accuracy: {acc * 100:.2f}%, Confidence: {conf:.4f}, AUROC: {auroc:.4f}") 