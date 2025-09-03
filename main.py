# main.py
# Code for "Beyond Static Bias: A Case for Dynamic, Per-Neuron Adaptation in Deep Networks"
# Author: Zrng Mahdi Tahir

import time
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

# ----------------- 1. DYNAMIC MULTI-BIAS LAYER -----------------
class DynamicMultiBiasLinear(nn.Module):
    def __init__(self, in_features, out_features, num_biases=8):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.biases = nn.Parameter(torch.Tensor(out_features, num_biases))
        self.gate_fc = nn.Linear(in_features, num_biases)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)); nn.init.zeros_(self.biases)
    def forward(self, x):
        main_output = F.linear(x, self.weight); alphas = F.softmax(self.gate_fc(x), dim=-1)
        bias_term = alphas @ self.biases.T; return main_output + bias_term

# ----------------- 2. THE NEW CNN LAYER: DYNAMICALLY MODULATED CONV -----------------
class DynamicModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_biases=4, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate_dmb = DynamicMultiBiasLinear(in_channels, out_channels, num_biases=num_biases)
    def forward(self, x):
        control_signal = self.pool(x).squeeze(-1).squeeze(-1)
        modulation = 1 + torch.tanh(self.gate_dmb(control_signal))
        x_conv = self.conv(x)
        return x_conv * modulation.unsqueeze(-1).unsqueeze(-1)

# ----------------- 3. CNN ARCHITECTURES (ResNet-like) -----------------
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        out += self.shortcut(x); out = F.relu(out); return out

class DMB_Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1, num_biases=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2_dmb = DynamicModulatedConv2d(planes, planes, kernel_size=3, padding=1, num_biases=num_biases)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2_dmb(out))
        out += self.shortcut(x); out = F.relu(out); return out

class ResNet(nn.Module):
    def __init__(self, block_builder, num_blocks, num_classes=10, num_biases=4):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.block_builder = lambda in_p, out_p, s: block_builder(in_p, out_p, s) if block_builder == BasicBlock else block_builder(in_p, out_p, s, num_biases)
        self.layer1 = self._make_layer(out_p=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layer(out_p=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layer(out_p=256, num_blocks=num_blocks[2], stride=2)
        self.linear = nn.Linear(256, num_classes)
    def _make_layer(self, out_p, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1); layers = []
        for s in strides:
            layers.append(self.block_builder(self.in_planes, out_p, s))
            self.in_planes = out_p
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.layer1(out); out = self.layer2(out); out = self.layer3(out)
        out = F.avg_pool2d(out, 8); out = out.view(out.size(0), -1); out = self.linear(out); return out

# ----------------- 4. TRAINING & BENCHMARKING FRAMEWORK -----------------
def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device):
    model.eval(); correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device); outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model, train_loader, test_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss(); optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); epoch_start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
            outputs = model(inputs); loss = criterion(outputs, labels); loss.backward(); optimizer.step()
        acc = evaluate(model, test_loader, device); duration = time.time() - epoch_start_time
        history.append({'epoch': epoch, 'accuracy': acc, 'duration': duration})
        print(f"Epoch {epoch:02d}: Test Acc = {acc*100:.2f}% | Time = {duration:.2f}s | LR = {scheduler.get_last_lr()[0]:.4f}")
        scheduler.step()
    return history

def run_cnn_benchmark(epochs=50, lr=0.1):
    print("="*50); print("🚀 Starting CNN SOTA-style Benchmark (CIFAR-10)"); print("="*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    num_biases_config = 4
    models_to_test = {
        "ResNet-S (Standard)": ResNet(BasicBlock, [2,2,2], num_biases=num_biases_config).to(device),
        f"ResNet-DMB (k={num_biases_config})": ResNet(DMB_Block, [2,2,2], num_biases=num_biases_config).to(device)
    }
    results = {}
    for name, model in models_to_test.items():
        print(f"\n--- Training: {name} | Params: {count_params(model):,} ---")
        history = train_model(model, train_loader, test_loader, epochs, lr, device)
        results[name] = history
    
    # Corrected data processing for the final summary table
    final_results = []
    for name, history in results.items():
        if history:
             best_acc = max(h['accuracy'] for h in history)
             avg_time = sum(h['duration'] for h in history) / len(history)
             params = count_params(models_to_test[name])
             final_results.append({
                 "Model": name,
                 "Best Accuracy": f"{best_acc*100:.2f}%",
                 "Avg. Epoch Time (s)": f"{avg_time:.2f}",
                 "Parameters": f"{params:,}"
             })

    df = pd.DataFrame(final_results)
    print("\n--- Final CNN Benchmark Results ---")
    print(df.to_string(index=False))

if __name__ == '__main__':
    run_cnn_benchmark(epochs=15) # NOTE: Reduced epochs for faster testing. Change to 50 for full run.
