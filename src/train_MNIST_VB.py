import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda import amp
from utils.datasets import load_MNIST
from utils.models import LeNetVB, set_seed


def train_vb(model, train_loader, epochs=100, lr=1e-3, tau=0.1):
    """
    Train LeNet-VB：
     - 1 MC sample per batch to compute output & KL
     - Loss = NLL + (tau / N) * KL
     - Adam + CosineAnnealingLR
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    num_data = len(train_loader.dataset)

    optimizer = optim.Adam(model.parameters(), lr=lr)
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


def evaluate_vb(model, test_loader, n_samples=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    import numpy as np
    from sklearn.metrics import roc_auc_score

    all_preds = []
    all_conf = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            outputs_stack = torch.zeros(n_samples, x.size(0), 10, device=device)
            for i in range(n_samples):
                logits, _ = model(x)
                probs = torch.softmax(logits, dim=1)
                outputs_stack[i] = probs

            mean_probs = outputs_stack.mean(dim=0)
            conf, preds = mean_probs.max(dim=1)
            all_preds.append(preds.cpu())
            all_conf.append(conf.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_conf = torch.cat(all_conf).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = 100.0 * (all_preds == all_labels).mean()
    correctness = (all_preds == all_labels).astype(int)
    auroc = roc_auc_score(correctness, all_conf)

    avg_conf = all_conf.mean()

    return accuracy, avg_conf, auroc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=10,
                        help='Tempering parameter τ for scaling KL by τ/N')
    parser.add_argument('--n_samples', type=int, default=15)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = load_MNIST(batch_size=args.batch_size)

    model = LeNetVB(num_classes=10, var0=1/33.0, estimator='flipout')

    print("Starting VB training (LeNet-VB)...")
    model = train_vb(model,
                     train_loader,
                     epochs=args.epochs,
                     lr=args.lr,
                     tau=args.tau)

    print("\nEvaluating VB model...")
    test_acc, test_conf, test_auroc = evaluate_vb(model, test_loader, n_samples=args.n_samples)
    print(f"Test Accuracy: {test_acc:.2f}%, "
          f"Avg Confidence: {test_conf:.4f}, "
          f"AUROC: {test_auroc:.4f}")

    torch.save(model.state_dict(), os.path.join(save_dir, 'MNIST_vb.pt'))
    print(f"Model weights saved to {save_dir}/MNIST_vb.pt")
