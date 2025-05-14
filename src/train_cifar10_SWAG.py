import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from utils.datasets import get_cifar10
from utils.models import CIFAR10CNN, SWAG
from utils.metrics import accuracy

def adjust_bn(model, loader, device):
    """
    Adjust batch normalization statistics using the entire loader
    """
    model.train()
    for images, _ in loader:
        images = images.to(device)
        model(images)

def train_swag(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader = get_cifar10(batch_size=args.batch_size)
    
    # Create model and optimizer
    base_model = CIFAR10CNN().to(device)
    if args.pretrained_path:
        base_model.load_state_dict(torch.load(args.pretrained_path))
    
    swag_model = SWAG(base_model, max_models=args.max_models).to(device)
    optimizer = optim.SGD(base_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Starting SWAG training with learning rate {args.lr}")
    for epoch in range(args.epochs):
        base_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = base_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}')
        
        # Update SWAG model at the end of each epoch
        swag_model.update_parameters(base_model)
        
        # Test the base model
        base_model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = base_model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        # Print epoch stats
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch}:')
        print(f'Training - Loss: {total_loss/len(train_loader):.4f}, '
              f'Accuracy: {100.*correct/total:.2f}%')
        print(f'Testing  - Loss: {test_loss/len(test_loader):.4f}, '
              f'Accuracy: {100.*test_correct/test_total:.2f}%')
        print(f'Time: {epoch_time:.2f}s')
        
        # Save model
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(swag_model.state_dict(), 
                  f'models/swag_cifar10_epoch_{epoch}.pt')
    
    print("Training finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SWAG training for CIFAR10')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate (default: 0.002)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--max-models', type=int, default=40,
                        help='maximum number of SWAG models (default: 40)')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='path to pretrained model')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    train_swag(args) 