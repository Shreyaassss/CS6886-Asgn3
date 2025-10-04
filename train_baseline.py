import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import trange, tqdm
import pandas as pd


# ===============================================
# 0. Custom MobileNetV2 Architecture 
# (Modified for CIFAR-10 with stride 1 in early layers)
# ===============================================

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # Stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4) # Kernel size 4 for 32x32 input
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ===========================
# 1. Mean/Std computation
# ===========================
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data = next(iter(loader))[0]
    mean = data.mean(dim=[0, 2, 3])
    std = data.std(dim=[0, 2, 3])
    return mean, std


# ===========================
# 2. Dataloaders w/ Simple Augmentation
# ===========================
def get_dataloaders(batch_size=128, num_workers=2):
    base_transform = transforms.ToTensor()
    raw_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=base_transform)

    mean, std = compute_mean_std(raw_trainset)
    print("Computed Mean:", mean.tolist())
    print("Computed Std:", std.tolist())

    # Retaining simple, faster transforms: RandomCrop and RandomHorizontalFlip
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True) 
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return trainloader, testloader

# ===========================
# 3. MixUp helper (Restored for high accuracy)
# ===========================
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ===========================
# 4. Train/Eval (Updated to include MixUp and Scheduler)
# ===========================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, use_mixup=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step() # Scheduler step is now inside the loop (per batch)

        running_loss += loss.item() * inputs.size(0)
        
        # Accuracy check (based on unmixed targets for simplicity)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }
    torch.save(state, path)


# ===========================
# 5. Main (Updated for high accuracy)
# ===========================
def main(args):
    # Ensure correct worker count based on system warning (2 is safer than 4)
    args.num_workers = min(args.num_workers, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Incase of Google Drive
    # drive.mount('/content/drive')
    # save_dir = "/content/drive/MyDrive/cifar_mobilenet_checkpoints"
    save_dir = "./baseline_local_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # Data
    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers)

    # Model: Using custom MobileNetV2
    model = MobileNetV2(num_classes=10) 
    print("Initializing custom MobileNetV2 from scratch.")
    model = model.to(device)

    # Loss & optimizer (Using Label Smoothing for stability)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)

    # Warmup + Cosine schedule (Necessary for high LR/MixUp stability)
    warmup_epochs = 5
    total_steps = len(trainloader) * args.epochs
    warmup_steps = len(trainloader) * warmup_epochs
    cosine_steps = total_steps - warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Decay from 1.0 (after warmup) to 0.0 over the remaining steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / cosine_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}
    
    print(f"Initial Test Acc (before training): {evaluate(model, testloader, criterion, device)[1]:.2f}%")

    # Training loop with tqdm progress bar
    for epoch in trange(1, args.epochs + 1, desc="Training", total=args.epochs):
        # Pass the scheduler into train_one_epoch
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, scheduler, device, use_mixup=True)
        test_loss, test_acc = evaluate(
            model, testloader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch,
                             path=os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth"))
            tqdm.write(f"\nEpoch [{epoch}/{args.epochs}] "
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
                        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # Saving history for analysis
    pd.DataFrame(history).to_csv("accuracy_history.csv", index=False)

# To run from jupyter notebook:
# if __name__ == "__main__":
#     from types import SimpleNamespace
    
#     # Using SimpleNamespace to mimic argparse functionality easily
#     args = SimpleNamespace(
#         epochs=100, 
#         batch_size=128,
#         lr=0.1,
#         # Defaulting to 2 to heed the system warning
#         num_workers=2, 
#         save_every=10
#     )

    # To run using command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=10)
    args = parser.parse_args()

    main(args)