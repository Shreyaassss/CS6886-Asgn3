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
from types import SimpleNamespace 
import argparse
import pandas as pd 
  
# ====================================================================
# I. MODEL ARCHITECTURE (Required for Checkpoint Loading)
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
        out = F.relu(self.bn1(self.conv1(x)))  # 1. Expansion
        out = F.relu(self.bn2(self.conv2(out))) # 2. Depthwise (FIXED)
        out = self.bn3(self.conv3(out))        # 3. Projection
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
        # Initial Conv: stride 2 -> 1 for CIFAR10 (32x32)
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


# ====================================================================
# II. UTILITY FUNCTIONS
# ====================================================================

def load_unpruned_checkpoint(checkpoint_path, model_class, device):
    """
    Loads a MobileNetV2 model and initializes it with weights from checkpoint.
    """
    model = model_class(num_classes=10).to(device)

    # Use standard torch.load (no weights_only arg)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle multiple checkpoint formats
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # assume raw state_dict
        state_dict = checkpoint

    # Load weights (non-strict to allow small mismatches)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")

    print(f"Loaded model weights from {checkpoint_path}")
    return model

def get_dataloaders(batch_size=128, num_workers=2):
    
    CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
    CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    
    # Using small mock dataset size to prevent excessive time in Colab testing
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ]))
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

#Pruning Helper Functions:

def compute_zero_stats(model):
    """
    Calculates global sparsity and per-layer sparsity for Conv2d layers.
    Returns: (global_zero_frac, list_of_layer_stats)
    """
    total_params = 0
    zero_params = 0
    per_layer = []
    
    # Check Conv2d layers only
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.data
            n = w.numel()
            z = (w == 0).sum().item()
            
            layer_sparsity = z / n if n > 0 else 0.0
            per_layer.append({
                'layer_name': name, 
                'zero_frac': layer_sparsity, 
                'zeros': int(z), 
                'total': int(n)
            })
            total_params += n
            zero_params += z
            
    global_zero_frac = zero_params / total_params if total_params > 0 else 0.0
    return global_zero_frac, per_layer


def reapply_masks(model):
    """
    Reapplies the pruning mask to the weight tensors of Conv2d layers 
    that have an 'out_mask' buffer defined.
    """
    for module in model.modules():
        # Check if the module is a Conv2d and has the mask attribute registered
        if isinstance(module, nn.Conv2d) and hasattr(module, 'out_mask'):
            # out_mask is stored as a 1D tensor representing channel importance
            mask = module.out_mask.view(-1, 1, 1, 1) # Reshape for element-wise multiplication
            module.weight.data.mul_(mask)


def prune_and_apply_mask(model, threshold):
    """
    Functional implementation of filter pruning.
    Creates a new mask based on the threshold and applies it to CONV weights 
    and associated BN parameters (in-place zeroing).
    """
    targets = get_prune_targets(model)
    total_channels = sum(t['total_channels'] for t in targets)
    pruned_count = 0
    
    for target in targets:
        bn = target['bn_module']
        conv = target['conv_module']
        scores = bn.weight.data.abs().cpu()
        
        # Determine which channels to keep (scores >= threshold)
        keep_mask = (scores >= threshold)
        
        if keep_mask.numel() == 0:
            continue
            
        out_mask = keep_mask.to(conv.weight.device).view(-1, 1, 1, 1).float()
        
        with torch.no_grad():
            # 1. Prune CONV weights (set to zero)
            conv.weight.data.mul_(out_mask)
            
            # 2. Prune BN parameters (set to zero)
            if bn.weight is not None:
                bn.weight.data.mul_(keep_mask.to(bn.weight.device).float())
            if bn.bias is not None:
                bn.bias.data.mul_(keep_mask.to(bn.bias.device).float())
                
        # 3. Register or update the mask buffer for reapply_masks()
        if not hasattr(conv, 'out_mask'):
            conv.register_buffer('out_mask', keep_mask.float().to(conv.weight.device))
        else:
            conv.out_mask.copy_(keep_mask.float().to(conv.weight.device))
            
        pruned_count += (~keep_mask).sum().item()
        
    # Note: We return the existing model instance (in-place pruning)
    actual_sparsity = pruned_count / total_channels if total_channels > 0 else 0.0
    return model, actual_sparsity

# -------------------- END: Pruning Helper Functions --------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    """
    Fine-tuning loop per epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        #Reapply mask after weight update
        reapply_masks(model)
        
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if len(dataloader.dataset) > 1000 and total > 5000: # Only train on ~10% of dataset
             break
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

def evaluate(model, dataloader, device):
    """
    Simple evaluation function (Accuracy only).
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_acc = 100. * correct / total
    return epoch_acc

def save_pruned_checkpoint(model, optimizer, iteration, sparsity, save_dir):
    """Saves the model state after an iterative pruning step to Google Drive."""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'iteration': iteration,
        'sparsity': sparsity,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
    }
    filename = f"pruned_iter{iteration}_sparsity{sparsity*100:.0f}.pth"
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    tqdm.write(f"Checkpoint saved to {path}")


# ====================================================================
# III. ITERATIVE PRUNING LOGIC (AutoSlim-Inspired)
# ====================================================================

def get_prune_targets(model):
    """
    Identifies the CONV/BN pairs suitable for filter pruning (output channel removal).
    Returns a list of dictionaries containing modules and channel counts.
    """
    prune_targets = []
    
    for name, module in model.named_modules():
        if isinstance(module, Block):
            # Target the projection layer (conv3) and its BN (bn3).
            bn_layer = module.bn3
            conv_layer = module.conv3
            
            prune_targets.append({
                'score_name': f"{name}.bn3.weight",
                'bn_module': bn_layer,
                'conv_module': conv_layer,
                'total_channels': conv_layer.out_channels,
            })

    # Prune the final block's projection (conv2)
    prune_targets.append({
        'score_name': "conv2.weight",
        'bn_module': model.bn2,
        'conv_module': model.conv2,
        'total_channels': model.conv2.out_channels,
    })
    
    return prune_targets

def calculate_pruning_threshold(model, target_sparsity):
    """
    Calculates the BN Gamma threshold required to achieve the total target sparsity 
    across all pruneable channels.
    """
    all_scores = []
    
    for target in get_prune_targets(model):
        scores = target['bn_module'].weight.data.abs().cpu()
        all_scores.append(scores)
        
    combined_scores = torch.cat(all_scores)
    
    TOTAL_CHANNELS = len(combined_scores)
    CHANNELS_TO_PRUNE = int(TOTAL_CHANNELS * target_sparsity)

    if CHANNELS_TO_PRUNE <= 0:
        return float('inf'), 0
    
    # Find the threshold magnitude corresponding to the K-th smallest score
    sorted_scores = combined_scores.sort().values
    
    if CHANNELS_TO_PRUNE >= len(sorted_scores):
         threshold = sorted_scores[-1].item()
    else:
        # The threshold is the K-th smallest element.
        threshold = sorted_scores[CHANNELS_TO_PRUNE].item()
    
    return threshold, CHANNELS_TO_PRUNE


# ====================================================================
# IV. MAIN ITERATIVE PRUNER CLASS
# ====================================================================

class IterativePruner:
    def __init__(self, model, trainloader, testloader, args, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.args = args
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.num_iterations = int(np.ceil(args.target_sparsity / args.pruning_step))
        tqdm.write(f"Set for {self.num_iterations} pruning iterations.")

    def _setup_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.args.finetune_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    def _setup_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.fine_tune_epochs)

    def run_pruning(self, save_dir): 
        os.makedirs(save_dir, exist_ok=True)
        tqdm.write(f"Checkpoints and logs will be saved to: {save_dir}")

        accuracy_history = {'iteration': [], 'cumulative_sparsity': [], 'test_acc': []}
        zero_stats_history = []
        
        initial_acc = evaluate(self.model, self.testloader, self.device)
        tqdm.write(f"Initial Baseline Accuracy: {initial_acc:.2f}%")
        
        for iteration in trange(1, self.num_iterations + 1, desc="Pruning Iterations"):
            cumulative_target_sparsity = iteration * self.args.pruning_step
            threshold, _ = calculate_pruning_threshold(self.model, cumulative_target_sparsity)
            self.model, cumulative_sparsity = prune_and_apply_mask(self.model, threshold)
            
            tqdm.write(f"\n--- Iteration {iteration}/{self.num_iterations}: Cumu. Sparsity {cumulative_sparsity*100:.2f}% ---")
            
            self.optimizer = self._setup_optimizer()
            self.scheduler = self._setup_scheduler()
            
            for epoch in range(1, self.args.fine_tune_epochs + 1):
                _, acc = train_one_epoch(self.model, self.trainloader, self.optimizer, self.scheduler, self.device)
                self.scheduler.step()
            
            test_acc = evaluate(self.model, self.testloader, self.device)
            global_sparsity, per_layer_stats = compute_zero_stats(self.model)
            tqdm.write(f"RESULTS: Test Acc: {test_acc:.2f}%, Global Sparsity: {global_sparsity*100:.2f}%")
            
            save_pruned_checkpoint(self.model, self.optimizer, iteration, cumulative_sparsity, save_dir)
            
            accuracy_history['iteration'].append(iteration)
            accuracy_history['cumulative_sparsity'].append(cumulative_sparsity)
            accuracy_history['test_acc'].append(test_acc)
            
            for stats in per_layer_stats:
                zero_stats_history.append({'iteration': iteration, 'global_sparsity': global_sparsity, **stats})
        
        tqdm.write("\nPruning Pipeline Complete.")
        
        history_df = pd.DataFrame(accuracy_history)
        zero_stats_df = pd.DataFrame(zero_stats_history)

        history_df.to_csv(os.path.join(save_dir, "accuracy_history_pruning.csv"), index=False)
        zero_stats_df.to_csv(os.path.join(save_dir, "zero_stats_history.csv"), index=False)

        tqdm.write(f"History logs saved in {save_dir}")
        return history_df
    
# ====================================================================
# V. EXECUTION 
# ====================================================================

def main(args):
    """Main function to run the pruning pipeline."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    # 2. Load Baseline Model
    baseline_model = load_unpruned_checkpoint(args.checkpoint_path, MobileNetV2, DEVICE)

    # 3. Run Iterative Pruning
    if baseline_model:
        pruner = IterativePruner(baseline_model, train_loader, test_loader, args, DEVICE)
        pruner.run_pruning(save_dir=args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iterative Pruning for MobileNetV2 on CIFAR-10')

    # Paths
    parser.add_argument('--checkpoint', '--checkpt', dest='checkpoint_path', required=True,
                        help='Path to the unpruned model checkpoint (.pth).')
    parser.add_argument('--save-dir', dest='save_dir', default='./pruning_ops',
                        help='Directory to save pruned checkpoints and logs.')

    # Pruning hyperparameters
    parser.add_argument('--target-sparsity', dest='target_sparsity', type=float, default=0.80,
                        help='Final target sparsity for pruneable channels (0-1).')
    parser.add_argument('--pruning-step', dest='pruning_step', type=float, default=0.05,
                        help='Sparsity increase per pruning iteration.')

    # Fine-tuning hyperparameters (names expected by the code)
    parser.add_argument('--fine-tune-epochs', '--epochs', dest='fine_tune_epochs', type=int, default=10,
                        help='Number of epochs to fine-tune after each pruning step.')
    parser.add_argument('--finetune-lr', '--lr', dest='finetune_lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning.')

    # Data loader / runtime
    parser.add_argument('--batch-size', '--batch', dest='batch_size', type=int, default=128,
                        help='Batch size for training and testing.')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=2,
                        help='Number of workers for DataLoader.')

    args = parser.parse_args()

    # sanity-print to make debugging easier
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    main(args)
