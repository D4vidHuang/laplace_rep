import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp
from utils.datasets import get_cifar10
from utils.models import WideResNetVB, set_seed


def train_vb(model, train_loader, epochs=100, lr=1e-3, tau=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    num_data = len(train_loader.dataset)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
    # T_max = total iterations = epochs * num_batches
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=epochs * len(train_loader),
                                                     eta_min=0)

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                outputs, kl = model(x)
                loss = F.cross_entropy(outputs.squeeze(), y) + tau / num_data*kl

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * x.size(0)
            preds = outputs.detach().argmax(dim=1)
            running_acc += (preds == y).sum().item()
            total += y.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * running_acc / total
        print(f'Epoch [{epoch+1}/{epochs}]  '
              f'Loss: {epoch_loss:.4f}  '
              f'Accuracy: {epoch_acc:.2f}%')
        print(f"[DEBUG] var0={model.conv1.prior_variance}, KL={kl.item():.4f}")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=3680,
                        help='Tempering parameter τ for scaling KL by τ/N')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)

    train_loader, test_loader = get_cifar10(batch_size=args.batch_size)

    model = WideResNetVB(16, 4, num_classes=10, var0=1/33.0)

    print("Starting VB training (WRN-VB)...")
    model = train_vb(model,
                     train_loader,
                     epochs=args.epochs,
                     lr=args.lr,
                     tau=args.tau)

    torch.save(model.state_dict(), os.path.join(save_dir, 'CIFAR10_vb.pt'))
    print(f"Model weights saved to {save_dir}/CIFAR10_vb.pt")
