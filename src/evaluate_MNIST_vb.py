import torch
import torch.nn.functional as F
from train_MNIST_VB import BayesianLeNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.datasets import get_mnist, get_ood_mnist
from utils.models import set_seed


def mc_forward(model, x, n_samples):
    probs = []
    for _ in range(n_samples):
        probs.append(F.softmax(model(x), 1))
    return torch.stack(probs, 0).mean(0)


def evaluate_vb_mnist(n_samples=20, ood_dataset='emnist'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained VB model
    model = BayesianLeNet().to(device)
    model.load_state_dict(torch.load('models/MNIST_vb.pt'))
    model.eval()
    
    # Data loading
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    ood_loader = get_ood_mnist(ood_dataset)

    def loop(loader, is_ood=False):
        all_conf, all_preds, all_labels = [], [], []
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                p = mc_forward(model, x, n_samples)
                conf, pred = p.max(1)

                all_conf.append(conf.cpu())
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())

                if not is_ood:
                    correct += pred.eq(y).sum().item()
                    total   += y.size(0)

        conf_arr  = torch.cat(all_conf).numpy()
        preds_cat = torch.cat(all_preds)
        labels_cat= torch.cat(all_labels)
        acc = None if is_ood else correct / total
        return acc, conf_arr, preds_cat, labels_cat

    acc_id, conf_id_arr, preds_id, labels_id = loop(test_loader, is_ood=False)
    _, conf_ood_arr, _, _ = loop(ood_loader, is_ood=True)

    mean_conf_id = conf_id_arr.mean()
    mean_conf_ood = conf_ood_arr.mean()

    bin_targets = preds_id.eq(labels_id).numpy().astype(int)
    id_auroc = roc_auc_score(bin_targets, conf_id_arr)

    all_scores = np.concatenate([conf_id_arr, conf_ood_arr])
    all_targets = np.concatenate([np.ones_like(conf_id_arr),
                                      np.zeros_like(conf_ood_arr)])
    ood_auroc = roc_auc_score(all_targets, all_scores)

    print(f"[VB] ID  Accuracy: {acc_id * 100:.2f}%  "
            f"ID-Conf: {mean_conf_id * 100:.4f}  "
            f"ID-AUROC: {id_auroc:.4f}")
    print(f"[VB] OOD={ood_dataset.upper():7s}  "
            f"Conf: {mean_conf_ood:.4f}  "
            f"OOD-AUROC: {ood_auroc:.4f}")

    return acc_id, mean_conf_id, mean_conf_ood, id_auroc, ood_auroc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=20,
                       help='Number of forward passes for uncertainty estimation')
    parser.add_argument('--ood_dataset', type=str, default='emnist',
                        help='OOD dataset')
    parser.add_argument('--seed', type=int, default=111,
                        help='Random seed')
    args = parser.parse_args()
    set_seed(args.seed)
    
    evaluate_vb_mnist(args.n_samples, args.ood_dataset)