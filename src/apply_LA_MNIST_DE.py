import torch
from laplace import Laplace
from utils.models import MLP, LeNet
from utils.datasets import get_mnist
import os

def apply_la_to_ensemble(model_name='lenet', la_type='la', num_models=5):
    """
    对Deep Ensemble中的每个模型应用LA或LA*
    
    Args:
        model_name: 使用的模型架构 ('mlp' 或 'lenet')
        la_type: LA的类型 ('la' 或 'la_star')
        num_models: Deep Ensemble中的模型数量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = get_mnist(batch_size=128)
    
    # 设置LA的类型
    if la_type == 'la':
        hessian = 'kron'
    elif la_type == 'la_star':
        hessian = 'full'
    else:
        raise ValueError("la_type must be 'la' or 'la_star'")
    
    # 对每个模型应用LA
    for i in range(num_models):
        print(f"\nApplying {la_type.upper()} to model {i+1}/{num_models}")
        
        # 加载MAP模型
        model = MLP() if model_name == 'mlp' else LeNet()
        map_model_path = f'models/MNIST_de_model_{i+1}.pt'
        model.load_state_dict(torch.load(map_model_path, map_location=device))
        model.to(device)
        
        # 应用LA
        la = Laplace(model=model,
                    likelihood='classification',
                    subset_of_weights='last_layer',
                    hessian_structure=hessian)
        la.fit(train_loader)
        la.optimize_prior_precision(method='marglik')
        
        # 保存LA模型
        save_path = f'models/MNIST_de_{la_type}_model_{i+1}.pt'
        os.makedirs('models', exist_ok=True)
        torch.save(la.state_dict(), save_path)
        print(f"Laplace approximation model {i+1} saved to {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'lenet'], default='lenet')
    parser.add_argument('--la_type', choices=['la', 'la_star'], default='la')
    parser.add_argument('--num_models', type=int, default=5,
                        help='Number of models in the Deep Ensemble')
    args = parser.parse_args()

    apply_la_to_ensemble(args.model, args.la_type, args.num_models) 