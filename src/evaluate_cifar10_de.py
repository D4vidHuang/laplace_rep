import argparse
import torch
import numpy as np
from utils.datasets import get_cifar10
from utils.models import CIFAR10CNN
from utils.metrics import evaluate, evaluate_la
from laplace import Laplace
from sklearn.metrics import roc_auc_score

def evaluate_ensemble(models, test_loader, eval_func=evaluate):
    """评估集成模型的性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_scores = []
    all_preds = []
    
    # 收集每个模型的预测
    for model in models:
        acc, conf, scores, targets, preds, labels, _ = eval_func(model, test_loader, ood=False)
        all_scores.append(scores)
        all_preds.append(preds)
    
    # 将所有模型的预测转换为numpy数组
    all_scores = np.stack(all_scores, axis=0)  # (n_models, n_samples)
    all_preds = np.stack(all_preds, axis=0)    # (n_models, n_samples)
    
    # 计算集成预测
    mean_scores = np.mean(all_scores, axis=0)  # (n_samples,)
    # 使用投票方式获取最终预测
    ensemble_preds = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=all_preds
    )
    
    # 计算准确率
    correct = (ensemble_preds == labels.cpu().numpy()).sum()
    total = len(labels)
    accuracy = correct / total
    
    # 计算平均置信度
    confidence = np.mean(mean_scores)
    
    # 计算AUROC
    binary_targets = (ensemble_preds == labels.cpu().numpy()).astype(int)
    auroc = roc_auc_score(binary_targets, mean_scores)
    
    return accuracy, confidence, auroc

def run_evaluation(mode='map', num_models=5, batch_size=128):
    """运行评估流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_cifar10(batch_size=batch_size)
    
    models = []
    for i in range(num_models):
        if mode == 'map':
            # 加载MAP模型
            model = CIFAR10CNN()
            model_path = f'models/CIFAR10_de_model_{i+1}.pt'
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            models.append(model)
            eval_func = evaluate
        else:  # la or la_star
            # 加载MAP模型
            model = CIFAR10CNN()
            model_path = f'models/CIFAR10_de_model_{i+1}.pt'
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            # 创建并加载LA模型
            hessian = 'kron' if mode == 'la' else 'full'
            la = Laplace(model,
                        likelihood='classification',
                        subset_of_weights='last_layer',
                        hessian_structure=hessian)
            la_path = f'models/CIFAR10_de_{mode}_model_{i+1}.pt'
            la.load_state_dict(torch.load(la_path, map_location=device))
            models.append(la)
            eval_func = evaluate_la
    
    # 评估整个集成
    acc, conf, auroc = evaluate_ensemble(models, test_loader, eval_func)
    return acc, conf, auroc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['map', 'la', 'la_star'], default='map',
                        help='Evaluation mode')
    parser.add_argument('--num_models', type=int, default=5,
                        help='Number of models in the ensemble')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for evaluation')
    args = parser.parse_args()

    acc, conf, auroc = run_evaluation(
        args.mode, args.num_models, args.batch_size
    )
    
    print(f"\nCIFAR10 Deep Ensemble [{args.mode.upper()}] Evaluation Results:")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Average Confidence: {conf:.4f}")
    print(f"AUROC: {auroc:.4f}") 