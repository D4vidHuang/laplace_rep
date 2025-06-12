# !!!This evaluation script is no longer used!!!
# base = SWAG(WideResNet(28, 10, num_classes=10).to(device)), ← this line of code creates a new SWAG model, while this
# is perfectly fine with LeNet (no BN), it will cause keyError with WRN (with BN). The simplest way of solving this is
# not to save the SWAG pack to disk and then load it in another script. So now the evaluation logic has been fully
# integrated into the train_cifar10_SWAG.py, the script will do training and evaluation in one go.

import argparse, time, torch, numpy as np
from sklearn.metrics import roc_auc_score

from utils.datasets import get_cifar10, get_ood_cifar10
from utils.models import WideResNet, set_seed
from utils.swag import predict_swag, SWAG

parser = argparse.ArgumentParser()
parser.add_argument('--pack', type=str, required=True,
                    help='swag_pack_seedX.pt produced by train_cifar10_SWAG.py')
parser.add_argument('--ood', choices=['SVHN','CIFAR100'],
                    default='SVHN')
parser.add_argument('--n-samples', type=int, default=15)
parser.add_argument('--seed', type=int, default=111)
args = parser.parse_args()
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pack = torch.load(args.pack, map_location=device)

# 1. load model
base = SWAG(WideResNet(28, 10, num_classes=10).to(device))
base.load_state_dict(pack['model_state'])
swag_samples = pack['samples']
swag_bn_params = pack['bn_params']

_, id_loader = get_cifar10(batch_size=512)
ood_loader = get_ood_cifar10(args.ood, batch_size=512)

def collect_conf(loader):
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

t0 = time.time()
conf_id, pred_id, lab_id = collect_conf(id_loader)
conf_ood, _, _ = collect_conf(ood_loader)

acc_id = pred_id.eq(lab_id).float().mean().item()
auroc_id = roc_auc_score(pred_id.eq(lab_id).numpy(), conf_id.numpy())
labels = np.concatenate([np.ones_like(conf_id), np.zeros_like(conf_ood)])
auroc_ood = roc_auc_score(labels, np.concatenate([conf_id, conf_ood]))

print(f"[SWG] Accuracy: {acc_id * 100:.2f}%")
print(f"[SWG] OOD Dataset: {args.ood.upper()} | Confidence: {conf_ood.mean()*100:.3f} | AUROC: {auroc_ood*100:.3f}")
