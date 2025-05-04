import torch
from laplace import Laplace
from utils.models import MLP, LeNet
from utils.datasets import get_mnist

def apply_la(model_name='mlp', la_type='la'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP() if model_name == 'mlp' else LeNet()
    model.load_state_dict(torch.load(f'models/map.pt', map_location=device))
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
    torch.save(la.state_dict(), f'models/{la_type}.pt')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp', 'lenet'], default='mlp')
    parser.add_argument('--la_type', choices=['la', 'la_star'], default='la')
    args = parser.parse_args()

    apply_la(args.model, args.la_type)