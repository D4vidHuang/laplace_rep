import torch
from laplace import Laplace
from utils.models import CIFAR10CNN
from utils.datasets import get_cifar10
import os
import argparse

def apply_la(la_type='la'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFAR10CNN()
    
    map_model_path = 'models/CIFAR10_map.pt'
    model.load_state_dict(torch.load(map_model_path, map_location=device))
    model.to(device)

    train_loader, _ = get_cifar10(batch_size=128)

    if la_type == 'la':
        hessian = 'kron'
    elif la_type == 'la_star':
        hessian = 'full'
    else:
        raise ValueError("la_type must be 'la' or 'la_star'")

    la = Laplace(model=model,
                 likelihood='classification',
                 subset_of_weights='last_layer',
                 hessian_structure=hessian)
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')
    
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'models/CIFAR10_{la_type}.pt'
    torch.save(la.state_dict(), save_path)
    print(f"Laplace approximation model saved to {save_path}")
    
    return la

def evaluate_la(la_model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            probs = la_model(x, pred_type='glm', link_approx='probit')
            preds = probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct / total * 100
    print(f"Laplace Test Accuracy: {acc:.2f}%")
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply Laplace Approximation to CIFAR10 model')
    parser.add_argument('--la_type', choices=['la', 'la_star'], default='la',
                        help='Laplace approximation type: la=Kron, la_star=Full')
    parser.add_argument('--eval', action='store_true', help='Evaluate after fitting')
    args = parser.parse_args()

    la_model = apply_la(la_type=args.la_type)
    
    if args.eval:
        _, test_loader = get_cifar10(batch_size=128)
        evaluate_la(la_model, test_loader) 