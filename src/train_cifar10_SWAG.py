import numpy as np
import argparse, os, time, torch
from utils.datasets import get_cifar10, get_ood_cifar10
from utils.models import WideResNet, set_seed
from utils.swag import fit_swag_and_precompute_bn_params, predict_swag, SWAG
from sklearn.metrics import roc_auc_score


def collect_conf(base, loader, swag_samples, swag_bn_params, device):
    conf_list, pred_list, label_list = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = predict_swag(base, x, swag_samples, swag_bn_params)
            conf, pred = p.max(1)
            conf_list.append(conf.cpu())
            pred_list.append(pred.cpu())
            label_list.append(y.cpu())
    return torch.cat(conf_list), torch.cat(pred_list), torch.cat(label_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-path', type=str, required=True,
                    help='path to pretrained MAP checkpoint (.pt)')
    parser.add_argument('--save-dir', type=str, default='models',
                    help='folder to save swag_cifar_*.pt')
    parser.add_argument('--seed', type=int, default=111)

    # paper settings
    parser.add_argument('--snapshots', type=int, default=40)
    parser.add_argument('--swg-lr', type=float, default=2e-3)
    parser.add_argument('--c-epochs', type=int, default=1)
    parser.add_argument('--c-batches', type=int, default=None)
    parser.add_argument('--bn-subset', type=float, default=1.0)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, id_loader = get_cifar10(batch_size=128)

    # 1. start from pretrained MAP
    base = WideResNet(16, 4, num_classes=10).to(device)
    base.load_state_dict(torch.load(args.map_path, map_location=device))

    # 2. train and bn
    t0 = time.time()
    swag_model, swag_samples, swag_bn = fit_swag_and_precompute_bn_params(
        model=base,
        device=device,
        train_loader=train_loader,
        max_num_models=args.snapshots,
        swg_lr=args.swg_lr,
        swg_c_epochs=args.c_epochs,
        swg_c_batches=args.c_batches,
        parallel=False,
        n_samples=30,
        bn_update_subset=args.bn_subset)

    print(f'SWAG finished in {(time.time()-t0):.1f}s')

    # 3. save
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f'swag_cifar_seed{args.seed}.pt')
    torch.save({'model_state': swag_model.state_dict(),
            'samples': swag_samples,
            'bn_params': swag_bn}, path)
    print(f'Saved to {path}')

    # 4. evaluate
    conf_id, pred_id, lab_id = collect_conf(swag_model, id_loader, swag_samples, swag_bn, device)
    acc_id = pred_id.eq(lab_id).float().mean().item()
    print(f"[SWG] Accuracy: {acc_id * 100:.2f}% | Confidence: {conf_id.mean() * 100:.3f}%")

    for ood in ['SVHN', 'CIFAR100']:
        print(f'Evaluating {ood}')
        ood_loader = get_ood_cifar10(ood, batch_size=128)
        conf_ood, _, _ = collect_conf(swag_model, ood_loader, swag_samples, swag_bn, device)
        labels = np.concatenate([np.ones_like(conf_id), np.zeros_like(conf_ood)])
        auroc_ood = roc_auc_score(labels, np.concatenate([conf_id, conf_ood]))

        print(
            f"[SWG] OOD Dataset: {ood} | Confidence: {conf_ood.mean() * 100:.3f} | AUROC: {auroc_ood * 100:.3f}")


if __name__ == '__main__':
    main()
