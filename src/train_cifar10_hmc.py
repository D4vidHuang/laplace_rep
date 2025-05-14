'''Train CIFAR10 with HMC implementation'''
from __future__ import print_function
import sys
import os

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from resources.csgmcmc.models import *
import numpy as np
import random

parser = argparse.ArgumentParser(description='HMC CIFAR10 Training')
parser.add_argument('--device_id', type=int, help='device id to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--temperature', type=float, default=1./50000,
                    help='temperature (default: 1/dataset_size)')
parser.add_argument('--num_samples', type=int, default=20,
                    help='number of HMC samples to collect')
parser.add_argument('--L', type=int, default=50,
                    help='number of leapfrog steps')
parser.add_argument('--epsilon', type=float, default=0.0001,
                    help='leapfrog step size')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Data and model directories
data_path = os.path.join(project_root, 'data')
model_dir = os.path.join(project_root, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
net = ResNet18()
if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True

def compute_gradient(net, inputs, targets, criterion):
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    return loss.item()

def leapfrog_step(net, momentum, epsilon, inputs, targets, criterion):
    # Half step for momentum
    for p, m in zip(net.parameters(), momentum):
        p.grad.data.zero_()
    loss = compute_gradient(net, inputs, targets, criterion)
    for p, m in zip(net.parameters(), momentum):
        m.data = m.data - 0.5 * epsilon * p.grad.data
    
    # Full step for position
    for p, m in zip(net.parameters(), momentum):
        p.data = p.data + epsilon * m.data
    
    # Half step for momentum
    for p, m in zip(net.parameters(), momentum):
        p.grad.data.zero_()
    loss = compute_gradient(net, inputs, targets, criterion)
    for p, m in zip(net.parameters(), momentum):
        m.data = m.data - 0.5 * epsilon * p.grad.data
    
    return loss

def hmc_step(net, inputs, targets, criterion, L, epsilon):
    # Initialize momentum
    momentum = []
    for p in net.parameters():
        m = torch.randn_like(p.data)
        if use_cuda:
            m = m.cuda(device_id)
        momentum.append(m)
    
    # Store initial state
    old_params = [p.data.clone() for p in net.parameters()]
    old_momentum = [m.data.clone() for m in momentum]
    
    # Initial Hamiltonian
    initial_loss = compute_gradient(net, inputs, targets, criterion)
    initial_kinetic = sum(0.5 * (m.data ** 2).sum() for m in momentum)
    initial_hamiltonian = initial_loss + initial_kinetic
    
    # Leapfrog integration
    current_loss = initial_loss
    for _ in range(L):
        current_loss = leapfrog_step(net, momentum, epsilon, inputs, targets, criterion)
    
    # Final Hamiltonian
    final_loss = current_loss
    final_kinetic = sum(0.5 * (m.data ** 2).sum() for m in momentum)
    final_hamiltonian = final_loss + final_kinetic
    
    # Metropolis-Hastings acceptance
    hamiltonian_delta = final_hamiltonian - initial_hamiltonian
    acceptance_prob = min(1.0, torch.exp(-hamiltonian_delta))
    
    if torch.rand(1) < acceptance_prob:
        return True, final_loss
    else:
        # Reject: restore old state
        for p, old_p in zip(net.parameters(), old_params):
            p.data = old_p
        return False, initial_loss

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    accepted_samples = 0
    total_samples = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
        
        accepted, loss = hmc_step(net, inputs, targets, criterion, args.L, args.epsilon)
        if accepted:
            accepted_samples += 1
        total_samples += 1
        
        train_loss += loss
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Accept Rate: %.3f%%'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total,
                   100.*accepted_samples/total_samples))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/len(testloader), correct, total, 100.*correct/total))
    return correct/total

criterion = nn.CrossEntropyLoss()

# Training loop
best_acc = 0
for epoch in range(args.epochs):
    train(epoch)
    acc = test(epoch)
    
    # Save model
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_dir, f'cifar10_hmc_epoch_{epoch}.pt'))
        best_acc = acc 