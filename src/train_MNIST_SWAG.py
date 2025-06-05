import argparse, os, time, torch
from utils.datasets import load_MNIST
from utils.models import LeNet, set_seed
from utils.swag import fit_swag_and_precompute_bn_params

parser = argparse.ArgumentParser()
parser.add_argument('--map-path', type=str, required=True,
                    help='path to pretrained MAP checkpoint (.pt)')
parser.add_argument('--save-dir', type=str, default='models',
                    help='folder to save swag_pack_*.pt')
parser.add_argument('--seed', type=int, default=111)

# paper settings
parser.add_argument('--snapshots', type=int, default=40)
parser.add_argument('--swa-lr',    type=float, default=1e-2)
parser.add_argument('--c-epochs',  type=int, default=1)
parser.add_argument('--c-batches', type=int, default=None)
parser.add_argument('--bn-subset', type=float, default=1.0)
args = parser.parse_args()
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, _, _ = load_MNIST(batch_size=128)

# 1. start from pretrained MAP
base = LeNet().to(device)
base.load_state_dict(torch.load(args.map_path, map_location=device))

# 2. train and bn
t0 = time.time()
swag_model, swag_samples, swag_bn = fit_swag_and_precompute_bn_params(
        model=base,
        device=device,
        train_loader=train_loader,
        max_num_models=args.snapshots,
        swa_lr=args.swa_lr,
        swa_c_epochs=args.c_epochs,
        swa_c_batches=args.c_batches,
        parallel=False,
        n_samples=30,
        bn_update_subset=args.bn_subset)

print(f'SWAG finished in {(time.time()-t0):.1f}s')

# 3. save
os.makedirs(args.save_dir, exist_ok=True)
path = os.path.join(args.save_dir, f'swag_pack_seed{args.seed}.pt')
torch.save({'model_state': swag_model.state_dict(),
            'samples'    : swag_samples,
            'bn_params'  : swag_bn}, path)
print(f'Saved to {path}')
