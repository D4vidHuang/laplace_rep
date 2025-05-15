import torch
from laplace import Laplace
from utils.models import GermanCreditMLP
from utils.datasets import get_german_credit
import os
import argparse

def apply_la(input_size=61, hidden_size=32, la_type='la'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GermanCreditMLP(input_size=input_size, hidden_size=hidden_size, num_classes=2)
    model.load_state_dict(torch.load(f'models/german_credit_map.pt', map_location=device))
    model.to(device)
    train_loader, _ = get_german_credit(batch_size=32)

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
    torch.save(la.state_dict(), f'models/german_credit_{la_type}.pt')
    print(f"Laplace approximation model saved to models/german_credit_{la_type}.pt")
    
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
    parser = argparse.ArgumentParser(description='Apply Laplace Approximation to German Credit model')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden layer size')
    parser.add_argument('--la_type', choices=['la', 'la_star'], default='la',
                        help='Laplace approximation type: la=Kron, la_star=Full')
    parser.add_argument('--eval', action='store_true', help='Evaluate after fitting')
    args = parser.parse_args()

    train_loader, test_loader = get_german_credit(batch_size=32)
    for x, _ in train_loader:
        input_size = x.size(1)
        print(f"Feature dimension: {input_size}")
        break
    
    la_model = apply_la(input_size=input_size, hidden_size=args.hidden_size, 
                         la_type=args.la_type)

    if args.eval:
        evaluate_la(la_model, test_loader) 