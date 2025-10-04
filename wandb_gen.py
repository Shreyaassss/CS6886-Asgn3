import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import wandb
from types import SimpleNamespace
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from google.colab import drive

# NOTE: Set the checkpoint directory path here
CHECKPOINT_DIR = "/content/drive/MyDrive/CS6886-Asgn3/pruned_checkpoints_80/pruning_checkpoints_80_v1" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_LR = 0.0001
FINE_TUNE_EPOCHS = 10
BATCH_SIZE = 128
NUM_WORKERS = 2

# ====================================================================
# I. ARCHITECTURE (The Custom MobileNetV2)
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
# II. QUANTIZATION & TRAINING UTILITIES (From Previous Context)
# ====================================================================

def linear_init(W_non_zero, num_bins):
    if W_non_zero.numel() == 0: return torch.tensor([]).to(W_non_zero.device)
    min_val, max_val = W_non_zero.min().item(), W_non_zero.max().item()
    return torch.linspace(min_val, max_val, num_bins).to(W_non_zero.device)

def quantize_weights(W_full, centroids, W_mask):
    if W_mask.sum() == 0: return W_full.data.new_zeros(W_full.shape), W_full.data.new_zeros(W_full.numel(), dtype=torch.long)
    W_non_zero = W_full[W_mask]
    distances = (W_non_zero.reshape(-1, 1) - centroids.reshape(1, -1)) ** 2
    indices_non_zero = torch.argmin(distances, dim=1)
    W_Q = W_full.clone().detach()
    W_Q[W_mask] = centroids[indices_non_zero]
    return W_Q, indices_non_zero

def update_centroids(model):
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, '_quant_W_full'):
                W_full, centroids, W_mask = module._quant_W_full.data, module._centroids.data, module._W_mask.data
                num_bins = module._centroids.numel()
                if W_mask.sum() == 0: continue
                _, indices_non_zero = quantize_weights(W_full, centroids, W_mask)
                W_non_zero = W_full[W_mask]
                new_centroids, counts = W_full.data.new_zeros(num_bins), W_full.data.new_zeros(num_bins)
                new_centroids.scatter_add_(0, indices_non_zero, W_non_zero)
                counts.scatter_add_(0, indices_non_zero, W_non_zero.new_ones(W_non_zero.size()))
                counts[counts == 0] = 1 
                module._centroids.copy_(new_centroids / counts)

def apply_quantization_to_model(model):
    for module in model.modules():
        if hasattr(module, '_quant_W_full'):
            W_full, centroids, W_mask = module._quant_W_full, module._centroids, module._W_mask
            W_Q, _ = quantize_weights(W_full.data, centroids.data, W_mask.data)
            W_Q_STE = W_full + (W_Q - W_full).detach()
            module._original_weight_data_ref = module.weight.data
            module.weight.data = W_Q_STE.data

def restore_full_precision_weights(model):
    for module in model.modules():
        if hasattr(module, '_original_weight_data_ref'):
            module.weight.data = module._original_weight_data_ref
            del module._original_weight_data_ref

def setup_quantization_attributes(model, conv_bit_width, fc_bit_width):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if not module.weight.requires_grad: continue
            bit_width = conv_bit_width if isinstance(module, nn.Conv2d) else fc_bit_width
            num_bins = 2**bit_width
            W_initial = module.weight.data
            W_mask = (W_initial != 0)
            W_non_zero = W_initial[W_mask]
            centroids_init = linear_init(W_non_zero, num_bins)
            W_full_param = nn.Parameter(W_initial.clone().detach().to(DEVICE))
            module.register_parameter('_quant_W_full', W_full_param)
            module.register_buffer('_centroids', centroids_init.to(DEVICE))
            module.register_buffer('_W_mask', W_mask.clone().detach().to(DEVICE))
            module.weight = module._quant_W_full 
    return model

def get_quantization_trainable_parameters(model):
    params = []
    for module in model.modules():
        if hasattr(module, '_quant_W_full'):
            params.append(module._quant_W_full)
    return params


def train_quantization_epoch(model, dataloader, optimizer, scheduler, device):
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

        if total > 5000: break # Partial dataset run for speed
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, device):
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

def calculate_compression_ratio(model, conv_bits, fc_bits):
    total_original_bits = 0
    total_quantized_bits = 0

    for module in model.modules():
        if hasattr(module, '_quant_W_full'):
            W = module._quant_W_full.data
            N_total = W.numel()
            total_original_bits += N_total * 32
            K = module._centroids.numel()
            B = K.bit_length() - 1 
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
# III. SWEEP DRIVER LOGIC
# ====================================================================

def get_dataloaders(batch_size=128, num_workers=2):
    CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
    CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def load_checkpoint_for_sweep(sparsity_data, model_class, device, checkpoint_dir):
    """Loads the specific checkpoint corresponding to the sparsity level."""
    iteration = int(sparsity_data['iteration'])
    sparsity_percent = int(round(sparsity_data['cumulative_sparsity'] * 100))
    
    if sparsity_percent == 0:
        # For 0% sparsity, use the highest accuracy baseline (Iteration 1)
        path = os.path.join(checkpoint_dir, "pruned_iter1_sparsity5.pth") 
    else:
        path = os.path.join(checkpoint_dir, f"pruned_iter{iteration}_sparsity{sparsity_percent}.pth")
        
    try:
        model = model_class(num_classes=10).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found for Iteration {iteration} at {path}. Skipping run.")
        return None
    except Exception as e:
        print(f"Error loading checkpoint at {path}: {e}")
        return None

def run_qat_for_sweep(sparsity_data, config, trainloader, testloader):
    """
    Executes the full 10-epoch K-means QAT fine-tuning for a single configuration.
    """
    
    # 1. Load Pruned Model
    model = load_checkpoint_for_sweep(sparsity_data, MobileNetV2, DEVICE, CHECKPOINT_DIR)
    if model is None:
        return None
    
    # 2. Setup Quantization (Adds _quant_W_full, _centroids, _W_mask)
    setup_quantization_attributes(model, config.conv_bits, config.fc_bits)
    
    # 3. Setup Trainer (Optimizers/Schedulers are built on the new parameters)
    trainer = SimpleNamespace(
        model=model, trainloader=trainloader, testloader=testloader, device=DEVICE
    )
    trainer.optimizer = optim.SGD(get_quantization_trainable_parameters(model), lr=BASE_LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    trainer.scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=FINE_TUNE_EPOCHS)
    
    # 4. Run QAT Fine-Tuning (10 epochs)
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        train_quantization_epoch(trainer.model, trainer.trainloader, trainer.optimizer, trainer.scheduler, trainer.device)
        update_centroids(trainer.model) # WCSS Minimization
        trainer.scheduler.step()
        
    # 5. Evaluate Final Accuracy and Metrics
    final_acc = evaluate(trainer.model, trainer.testloader, trainer.device)
    ratio, _, quantized_mb = calculate_compression_ratio(trainer.model, config.conv_bits, config.fc_bits)
    
    return final_acc, ratio, quantized_mb


def run_wandb_sweep():
    
    # 1. Load Data and Checkpoints
    try:
        pruning_df = pd.read_csv("accuracy_history_pruning.csv")
    except FileNotFoundError:
        print("ERROR: 'accuracy_history_pruning.csv' not found.")
        return

    # Filter to 0%, 25%, 50% sparsity levels
    pruning_df['sparsity_percent'] = (pruning_df['cumulative_sparsity'] * 100).round(0)
    target_sparsity_levels = [0, 25, 50]
    data_points = []
    for target in target_sparsity_levels:
        closest_row = pruning_df.iloc[(pruning_df['sparsity_percent'] - target).abs().argsort()[:1]]
        if not closest_row.empty:
            data_points.append(closest_row.iloc[0])

    if not data_points:
         print("ERROR: Could not isolate target sparsity levels (0%, 25%, 50%).")
         return
         
    # Load Dataloaders once
    trainloader, testloader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

    # 2. WANDB INITIALIZATION
    wandb.init(
        project="MobileNetV2-Compression-Sweep-v2",
        name="Assignment-Q3",
    )
    
    # 3. DEFINE QUANTIZATION SWEEP CONFIGURATIONS
    # Use 8 for non-quantized baseline
    conv_bits_levels = [8, 5, 4] 
    # Use 8 for non-quantized baseline
    fc_bits_levels = [8, 4, 3] 

    # 4. RUN THE SWEEP LOOP
    
    print("--- Starting ACTUAL QAT Compression Sweep ---")
    
    for sparsity_data in data_points:
        s_percent = sparsity_data['cumulative_sparsity'] * 100
        pruned_acc = sparsity_data['test_acc']
        
        for bc in conv_bits_levels:
            for bf in fc_bits_levels:
                config = SimpleNamespace(conv_bits=bc, fc_bits=bf) 
                
                print(f"RUNNING: Sparsity={s_percent:.0f}%, Bits={bc}/{bf} (Base Acc: {pruned_acc:.2f}%)")

                results = run_qat_for_sweep(sparsity_data, config, trainloader, testloader)
                
                if results:
                    final_acc, ratio, model_mb = results
                    
                    # Log the result to WandB
                    wandb.log({
                        "sparsity_percent": s_percent,
                        "weight_quant_bits_conv": bc,
                        "weight_quant_bits_fc": bf,
                        "compression_ratio": ratio,
                        "model_size_mb": model_mb,
                        "quantized_acc": final_acc,
                        "activation_quant_bits": 32, # Activations are still FP32
                    })
                    
                    print(f"  -> FINAL ACC: {final_acc:.2f}%, RATIO: {ratio:.2f}X, SIZE: {model_mb:.2f}MB")
                else:
                    print("  -> RUN SKIPPED DUE TO CHECKPOINT ERROR.")
        
    wandb.finish()
    print("--- WandB sweep logged successfully. Proceed to the Parallel Coordinates chart. ---")


if __name__ == '__main__':
    
    # NOTE: Ensure you are logged into wandb and that the CHECKPOINT_DIR is correct.
    run_wandb_sweep()
