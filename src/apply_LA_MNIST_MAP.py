import torch
from laplace import Laplace
from utils.models import MLP, LeNet
from utils.datasets import get_mnist
import os

def apply_la(model_name='mlp', la_type='la'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP() if model_name == 'mlp' else LeNet()
    
    # 使用新的命名格式加载MAP模型
    map_model_path = f'models/MNIST_map.pt'
    model.load_state_dict(torch.load(map_model_path, map_location=device))
    model.to(device)
    train_loader, _ = get_mnist(batch_size=128)

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
    
    # 使用新的命名格式保存LA模型
    save_path = f'models/MNIST_{la_type}.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(la.state_dict(), save_path)
    print(f"Laplace approximation model saved to {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'lenet'], default='lenet')
    parser.add_argument('--la_type', choices=['la', 'la_star'], default='la')
    args = parser.parse_args()

    apply_la(args.model, args.la_type)