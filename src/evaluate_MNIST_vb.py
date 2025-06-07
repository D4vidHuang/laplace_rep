import argparse, torch, numpy as np
from sklearn.metrics import roc_auc_score
from utils.datasets import load_MNIST, get_ood_mnist
from utils.models import LeNetVB, set_seed


@torch.no_grad()
def collect_conf_loader(model, loader, n_samples=15):
    """
    n_samples times MC：
      - return mean_confidence (np.array) & predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    conf_list = []
    correct_list = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # n_samples forward + softmax  → (n_samples, batch_size, num_classes)
        outputs = torch.zeros(n_samples, x.size(0), 10, device=device)
        for i in range(n_samples):
            logits, _ = model(x)
            outputs[i] = torch.softmax(logits, dim=1)

        mean_probs = outputs.mean(dim=0)
        conf, preds = mean_probs.max(dim=1)

        conf_list.append(conf.cpu())
        correct_list.append((preds == y).cpu())

    all_conf = torch.cat(conf_list).numpy()
    all_correct = torch.cat(correct_list).numpy()

    return all_conf, all_correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-path', type=str, required=True,
                        help='path to saved VB model weights')
    parser.add_argument('--n_samples', type=int, default=15)
    parser.add_argument('--ood', type=str, choices=['emnist','fmnist','kmnist'],
                        help='optional OOD dataset')
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    set_seed(args.seed)

    model = LeNetVB(num_classes=10, var0=1/33.0, estimator='flipout')
    state = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(state)

    # 2. ID
    _, _, id_loader = load_MNIST(batch_size=128)

    id_conf, id_corr = collect_conf_loader(model, id_loader, args.n_samples)
    id_acc = 100.0 * id_corr.mean()
    id_mean_conf = id_conf.mean()
    id_auroc = roc_auc_score(id_corr.astype(int), id_conf)

    print(f"[VB] seed={args.seed} | ID Accuracy: {id_acc:.2f}% | "
          f"ID-conf: {id_mean_conf:.4f} | ID-AUROC: {id_auroc:.4f}")

    # 3. OOD
    if args.ood:
        ood_loader = get_ood_mnist(args.ood, batch_size=128)
        ood_conf, _ = collect_conf_loader(model, ood_loader, args.n_samples)

        labels = np.concatenate([np.ones_like(id_conf), np.zeros_like(ood_conf)])
        scores = np.concatenate([id_conf, ood_conf])
        ood_auroc = roc_auc_score(labels, scores)
        ood_mean_conf = ood_conf.mean()

        print(f"[VB] OOD={args.ood.upper()} | OOD-conf: {ood_mean_conf:.4f} | "
              f"OOD-AUROC: {ood_auroc:.4f}")
