#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import trange, tqdm
import pandas as pd
import argparse 

# ====================================================================
# I. MODEL ARCHITECTURE
# ====================================================================

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
    cfg = [(1,  16, 1, 1), (6,  24, 2, 1), (6,  32, 3, 2), (6,  64, 4, 2), (6,  96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
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
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ====================================================================
# II. PROCEDURAL QUANTIZATION HELPERS
# ====================================================================

def linear_init(W_non_zero, num_bins):
    """Initializes centroids linearly based on the min/max of the non-zero weights."""
    if W_non_zero.numel() == 0:
        return torch.tensor([]).to(W_non_zero.device)
    min_val = W_non_zero.min().item()
    max_val = W_non_zero.max().item()
    centroids = torch.linspace(min_val, max_val, num_bins).to(W_non_zero.device)
    return centroids

def quantize_weights(W_full, centroids, W_mask):
    """
    K-means Assignment Step (Procedural Forward Pass).
    Assigns each non-zero weight in W_full to the nearest centroid.
    """
    if W_mask.sum() == 0:
        return W_full.data.new_zeros(W_full.shape), W_full.data.new_zeros(W_full.numel(), dtype=torch.long)
    W_non_zero = W_full[W_mask]
    W_flat = W_non_zero.reshape(-1, 1)
    C_flat = centroids.reshape(1, -1)
    distances = (W_flat - C_flat) ** 2
    indices_non_zero = torch.argmin(distances, dim=1)
    W_Q = W_full.clone().detach()
    W_Q[W_mask] = centroids[indices_non_zero]
    return W_Q, indices_non_zero

def update_centroids(model):
    """
    WCSS Minimization Step: Updates centroids based on the mean of assigned weights.
    Iterates over standard Conv2d/Linear modules using attached attributes.
    """
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, '_quant_W_full'):
                W_full = module._quant_W_full.data
                centroids = module._centroids.data
                W_mask = module._W_mask.data
                num_bins = module._centroids.numel()
                W_non_zero = W_full[W_mask]
                _, indices_non_zero = quantize_weights(W_full, centroids, W_mask)
                new_centroids = W_full.data.new_zeros(num_bins)
                counts = W_full.data.new_zeros(num_bins)
                new_centroids.scatter_add_(0, indices_non_zero, W_non_zero)
                counts.scatter_add_(0, indices_non_zero, W_non_zero.new_ones(W_non_zero.size()))
                counts[counts == 0] = 1
                module._centroids.copy_(new_centroids / counts)
    tqdm.write("Centroids updated via WCSS minimization (Mean-Update Rule).")

def calculate_compression_ratio(model, conv_bits, fc_bits):
    """
    Calculates the compression ratio based on the Deep Compression formula (sparse + quantization).
    Ratio = (Original 32-bit Float Size) / (Quantized Index Size + Codebook Size)
    """
    total_original_bits = 0
    total_quantized_bits = 0
    for name, module in model.named_modules():
        if hasattr(module, '_quant_W_full'):
            W = module._quant_W_full.data
            N_total = W.numel()
            total_original_bits += N_total * 32
            B = module._centroids.numel().bit_length() - 1
            K = module._centroids.numel()
            W_mask = module._W_mask.data
            N_sparse = W_mask.sum().item()
            indices_size = N_sparse * B
            codebook_size = K * 32
            total_quantized_bits += indices_size + codebook_size
    compression_ratio = total_original_bits / total_quantized_bits if total_quantized_bits > 0 else 1.0
    original_mb = total_original_bits / 8e6
    quantized_mb = total_quantized_bits / 8e6
    return compression_ratio, original_mb, quantized_mb

# ====================================================================
# III. MANUAL WEIGHT MANAGEMENT
# ====================================================================

def apply_quantization_to_model(model):
    """MANUAL PRE-HOOK: Quantizes weights and applies STE."""
    for module in model.modules():
        if hasattr(module, '_quant_W_full'):
            W_full = module._quant_W_full
            centroids = module._centroids
            W_mask = module._W_mask
            W_Q, _ = quantize_weights(W_full.data, centroids.data, W_mask.data)
            W_Q_STE = W_full + (W_Q - W_full).detach()
            module._original_weight_data_ref = module.weight.data
            module.weight.data = W_Q_STE.data

def restore_full_precision_weights(model):
    """MANUAL POST-HOOK: Restores original full-precision weights."""
    for module in model.modules():
        if hasattr(module, '_original_weight_data_ref'):
            module.weight.data = module._original_weight_data_ref
            del module._original_weight_data_ref

def setup_quantization_attributes(model, conv_bit_width, fc_bit_width):
    """Attaches parameters and buffers to layers for quantization."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if not module.weight.requires_grad:
                continue
            bit_width = conv_bit_width if isinstance(module, nn.Conv2d) else fc_bit_width
            num_bins = 2**bit_width
            W_initial = module.weight.data
            W_mask = (W_initial != 0)
            W_non_zero = W_initial[W_mask]
            centroids_init = linear_init(W_non_zero, num_bins)
            W_full_param = nn.Parameter(W_initial.clone().detach())
            module.register_parameter('_quant_W_full', W_full_param)
            module.register_buffer('_centroids', centroids_init)
            module.register_buffer('_W_mask', W_mask.clone().detach())
            module.weight = module._quant_W_full
    return model

def get_quantization_trainable_parameters(model):
    """Returns only the custom _quant_W_full parameters for the optimizer."""
    params = []
    for name, param in model.named_parameters():
        if name.endswith('.weight'):
            module_name = name.rsplit('.', 1)[0]
            try:
                module = model
                for part in module_name.split('.'):
                    module = getattr(module, part)
                if hasattr(module, '_centroids'):
                    params.append(param)
            except AttributeError:
                continue
    return params

# ====================================================================
# IV. TRAINING & EVALUATION
# ====================================================================

def load_pruned_checkpoint(checkpoint_path, model_class, device):
    """Loads the pruned model from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    model = model_class(num_classes=10).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    tqdm.write(f"Loaded pruned model weights from {checkpoint_path}")
    return model

def get_dataloaders(batch_size=128, num_workers=2):
    """Provides the standard CIFAR-10 dataloaders."""
    CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
    CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def train_quantization_epoch(model, dataloader, optimizer, device):
    """Fine-tuning loop for one epoch."""
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        apply_quantization_to_model(model)
        outputs = model(inputs)
        restore_full_precision_weights(model)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, device):
    """Simple evaluation function (Accuracy only)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            apply_quantization_to_model(model)
            outputs = model(inputs)
            restore_full_precision_weights(model)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    epoch_acc = 100. * correct / total
    return epoch_acc


class QuantizationTrainer:
    def __init__(self, model, trainloader, testloader, args, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.args = args
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

    def _setup_optimizer(self):
        trainable_params = get_quantization_trainable_parameters(self.model)
        optimizer = optim.SGD(trainable_params, lr=self.args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        return optimizer

    def _setup_scheduler(self):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
        return scheduler

    def run_quantization(self):
        save_dir = self.args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        tqdm.write(f"Results will be saved to: {save_dir}")
        history = {'epoch': [], 'test_acc': [], 'compression_ratio': []}
        
        initial_acc = evaluate(self.model, self.testloader, self.device)
        compression_ratio, original_mb, quantized_mb = calculate_compression_ratio(
            self.model, self.args.conv_bits, self.args.fc_bits
        )
        tqdm.write(f"Baseline (Pruned, Quantized Init) Accuracy: {initial_acc:.2f}%")
        tqdm.write(f"Initial Compression Ratio (Bits: {self.args.conv_bits}/{self.args.fc_bits}): {compression_ratio:.2f}X")
        tqdm.write(f"Model Size: {original_mb:.2f}MB (Original Float32) -> {quantized_mb:.2f}MB (Quantized)")
        
        for epoch in trange(1, self.args.epochs + 1, desc="Quantization Fine-Tuning"):
            loss, acc = train_quantization_epoch(self.model, self.trainloader, self.optimizer, self.device)
            update_centroids(self.model)
            self.scheduler.step()
            test_acc = evaluate(self.model, self.testloader, self.device)
            current_ratio, _, _ = calculate_compression_ratio(self.model, self.args.conv_bits, self.args.fc_bits)
            tqdm.write(f"Epoch {epoch}: Loss {loss:.4f} | Test Acc: {test_acc:.2f}% | Ratio: {current_ratio:.2f}X")
            history['epoch'].append(epoch)
            history['test_acc'].append(test_acc)
            history['compression_ratio'].append(current_ratio)

        tqdm.write("\nQuantization Fine-Tuning Complete.")
        
        final_path = os.path.join(save_dir, "final_quantized_model.pth")
        torch.save(self.model.state_dict(), final_path)
        tqdm.write(f"Final Quantized Model saved to {final_path}")
        
        history_df = pd.DataFrame(history)
        history_path = os.path.join(save_dir, "quantization_history.csv")
        history_df.to_csv(history_path, index=False)
        tqdm.write(f"History saved to {history_path}")
        return history_df

# ====================================================================
# V. EXECUTION
# ====================================================================
def main(args):
    """Main execution function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)
    
    # 2. Load the Pruned Baseline Model
    baseline_model = load_pruned_checkpoint(args.checkpoint, MobileNetV2, device)
    
    # 3. Setup Procedural Quantization
    model_quantized = setup_quantization_attributes(
        baseline_model, args.conv_bits, args.fc_bits
    )
    
    # 4. Run Quantization Fine-Tuning
    trainer = QuantizationTrainer(model_quantized, train_loader, test_loader, args, device)
    trainer.run_quantization()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNetV2 Quantization Training Script")
    
    # Quantization arguments
    parser.add_argument('--conv', type=int, default=2, help='Bit width for CONV layer quantization (2^k bins)')
    parser.add_argument('--fc', type=int, default=1, help='Bit width for FC layer quantization (2^k bins)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for fine-tuning')
    
    # Data arguments
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training and testing')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for dataloaders')
    
    # Path arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pruned model checkpoint')
    parser.add_argument('--save-dir', type=str, default='./quantization_ops', help='Directory to save results')

    args = parser.parse_args()
    main(args)
