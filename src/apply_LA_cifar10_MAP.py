import torch
from laplace import Laplace
from laplace.curvature import BackPackGGN, BackPackEF
from utils.models import WideResNet, set_seed
from utils.datasets import get_cifar10
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def apply_la(la_type='la'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet(16, 4, num_classes=10)

    # 使用新的命名格式加载MAP模型
    map_model_path = f'models/CIFAR10_map.pt'
    model.load_state_dict(torch.load(map_model_path, map_location=device))
    model.to(device)
    train_loader, _ = get_cifar10(batch_size=128)

    if la_type == 'la':
        hessian = 'kron'
        backend = BackPackGGN
    elif la_type == 'la_star':
        hessian = 'full'
        backend = BackPackEF
    else:
        raise ValueError("la_type must be 'la' or 'la_star'")

    la = Laplace(model=model,
                 likelihood='classification',
                 subset_of_weights='last_layer',
                 hessian_structure=hessian,
                 backend=backend)
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')

    # 使用新的命名格式保存LA模型
    save_path = f'models/CIFAR10_{la_type}.pt'
    os.makedirs('models', exist_ok=True)
    torch.save(la.state_dict(), save_path)
    print(f"Laplace approximation model saved to {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--la_type', choices=['la', 'la_star'], default='la')
    parser.add_argument('--seed', type=int, default='111')
    args = parser.parse_args()
    set_seed(args.seed)

    apply_la(args.la_type)
