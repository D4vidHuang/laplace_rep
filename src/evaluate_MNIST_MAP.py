import argparse

import torch
import numpy as np
from utils.datasets import get_mnist, get_ood_mnist
from utils.models import MLP, LeNet
from utils.metrics import evaluate, evaluate_la
from laplace import Laplace
from laplace.curvature import BackPackGGN, BackPackEF
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def run_evaluation(model_name='lenet', mode='map', ood_dataset='emnist', batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP() if model_name == 'mlp' else LeNet()
    
    # 使用新的命名格式加载模型
    map_model_path = 'models/MNIST_map.pt'
    model.load_state_dict(torch.load(map_model_path, map_location=device))
    model.to(device)

    if mode == 'map':
        eval_func = evaluate
        eval_model = model
    elif mode in ['la', 'la_star']:
        hessian = 'kron' if mode == 'la' else 'full'
        backend = BackPackGGN if mode == 'la' else BackPackEF
        la = Laplace(model,
                     likelihood='classification',
                     subset_of_weights='last_layer',
                     hessian_structure=hessian,
                     backend=backend)
        # 使用新的命名格式加载LA模型
        la_model_path = f'models/MNIST_{mode}.pt'
        la.load_state_dict(torch.load(la_model_path, map_location=device))
        eval_func = evaluate_la
        eval_model = la

    # 加载MNIST测试数据
    _, test_loader = get_mnist(batch_size=batch_size)
    
    # 评估模型在MNIST上的性能
    acc, conf, scores, targets, preds, labels = eval_func(eval_model, test_loader, ood=False)
    
    # 创建二分类问题来计算AUROC
    # 将预测正确的样本标记为正类，预测错误的样本标记为负类
    binary_targets = (preds == labels).cpu().numpy().astype(int)
    binary_scores = scores  # 使用模型置信度作为预测分数
    
    # 计算AUROC
    auroc = roc_auc_score(binary_targets, binary_scores)

    # OOD数据评估
    ood_loader = get_ood_mnist(ood_dataset, batch_size=batch_size)
    ood_acc, ood_conf, ood_scores, ood_targets, _, _ = eval_func(eval_model, ood_loader, ood=True)

    # OOD检测 AUROC
    all_scores = np.concatenate([scores, ood_scores])
    all_targets = np.concatenate([targets, ood_targets])
    ood_auroc = roc_auc_score(all_targets, all_scores)

    return acc, conf, auroc, ood_acc, ood_conf, ood_auroc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'lenet'], default='lenet')
    parser.add_argument('--mode', choices=['map', 'la', 'la_star'], default='map')
    parser.add_argument('--ood', choices=['emnist', 'fmnist', 'kmnist'], default='emnist')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for evaluation')
    args = parser.parse_args()

    acc, conf, auroc, ood_acc, ood_conf, ood_auroc = run_evaluation(args.model, args.mode, args.ood, args.batch_size)

    print(f"[{args.mode.upper()}] Accuracy: {acc * 100:.2f}% | Confidence: {conf*100:.3f} | AUROC: {auroc*100:.3f}")
    print(f"[{args.mode.upper()}] OOD Dataset: {args.ood.upper()} | Accuracy: {ood_acc * 100:.2f}% | Confidence: {ood_conf*100:.3f} | AUROC: {ood_auroc*100:.3f}")
 