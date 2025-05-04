import torch
from laplace import Laplace
import os

def apply_laplace(model, train_loader, model_dir, model_name, la_type='la'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    hessian = 'kron' if la_type == 'la' else 'full'
    la = Laplace(
        model=model,
        likelihood='classification',
        subset_of_weights='last_layer',
        hessian_structure=hessian
    )

    try:
        la.fit(train_loader)
        la.optimize_prior_precision(method='marglik')
        save_path = os.path.join(model_dir, f'{model_name}_{la_type}.pt')
        torch.save(la.state_dict(), save_path)
        print(f"Saved Laplace model to {save_path}")
        return la
    except Exception as e:
        raise RuntimeError(f"Failed to apply Laplace ({la_type}): {str(e)}")