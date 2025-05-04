import argparse

import torch
import numpy as np
from utils.datasets import get_mnist, get_ood_mnist
from utils.models import MLP, LeNet
from utils.metrics import evaluate, evaluate_la, fpr95, visualize
from laplace import Laplace
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'lenet'], default='mlp')
    parser.add_argument('--ood', choices=['emnist', 'fmnist', 'kmnist'], default='emnist')
    parser.add_argument('--mode', choices=['map', 'la', 'la_star'], default='map')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP() if args.model == 'mlp' else LeNet()
    model.load_state_dict(torch.load('models/map.pt', map_location=device))
    model.to(device)

    if args.mode == 'map':
        eval_func = evaluate
        eval_model = model
    elif args.mode in ['la', 'la_star']:
        hessian = 'kron' if args.mode == 'la' else 'full'
        la = Laplace(model,
                     likelihood='classification',
                     subset_of_weights='last_layer',
                     hessian_structure=hessian)
        la.load_state_dict(torch.load(f'models/{args.mode}.pt', map_location=device))
        eval_func = evaluate_la
        eval_model = la

    id_loader, _ = get_mnist(batch_size=128)
    ood_loader = get_ood_mnist(args.ood)

    id_acc, id_conf, id_scores, id_targets, id_preds, id_labels, id_images = eval_func(eval_model, id_loader, ood=False)
    ood_acc, ood_conf, ood_scores, ood_targets, _, _, _ = eval_func(eval_model, ood_loader, ood=True)

    all_scores = np.concatenate([id_scores, ood_scores])
    all_targets = np.concatenate([id_targets, ood_targets])
    auroc = roc_auc_score(all_targets, all_scores)
    fpr95 = fpr95(all_targets, all_scores)

    print(f"[{args.mode.upper()}] ID Accuracy: {id_acc * 100:.2f}%, Conf: {id_conf:.4f}")
    print(f"[{args.mode.upper()}] OOD Accuracy: {ood_acc * 100:.2f}%, Conf: {ood_conf:.4f}")
    print(f"[{args.mode.upper()}] AUROC: {auroc:.4f}, FPR@95: {fpr95:.4f}")

    visualize(id_images[:10], id_preds[:10], id_labels[:10], title=f"{args.mode.upper()} - ID Samples with Predictions")