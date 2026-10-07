"""
models_2d.py
Defines the edge-optimized deep learning physical-layer spectrum sensing architectures:
1. STFT-RADN (Residual Dense + CBAM Attention): Custom ~852.6k parameter network.
   Features CBAM (Channel & Spatial Attention) modules, Residual Dense Blocks (RDB),
   AdaptiveAvgPool2d, and a 5-class linear classifier.
2. ResNet18_2D: Adapted 1-channel baseline for comparison on 2D PSD spectrograms [Batch, 1, 32, 33].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. CBAM Attention Modules
# ==========================================

class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM."""
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM."""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining Channel and Spatial Attention."""
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ==========================================
# 2. Residual Dense Block (RDB)
# ==========================================

class RDB_Conv(nn.Module):
    """Single Convolutional Layer inside Residual Dense Block."""
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    """Residual Dense Block with dense feature reuse and local feature fusion."""
    def __init__(self, in_channels, num_layers=3, growth_rate=40):
        super().__init__()
        layers = []
        curr_channels = in_channels
        for _ in range(num_layers):
            layers.append(RDB_Conv(curr_channels, growth_rate))
            curr_channels += growth_rate
        self.layers = nn.ModuleList(layers)
        self.lff = nn.Sequential(
            nn.Conv2d(curr_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.lff(out)
        return out + x  # Residual skip connection


# ==========================================
# 3. STFT-RADN Main Model
# ==========================================

class STFT_RADN(nn.Module):
    """
    STFT-RADN: Residual Dense + CBAM Attention Network.
    
    Target Specs:
    - Input shape: [Batch, 1, 32, 33]
    - Parameter count: ~852,593 parameters
    - Output: Logits for 5 spectrum classes
    """
    def __init__(self, num_classes=5, channels=104, growth_rate=40, num_blocks=3):
        super().__init__()
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Dense Attention Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'rdb': RDB(channels, num_layers=3, growth_rate=growth_rate),
                'cbam': CBAM(channels),
                'conv': nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True)
                )
            }))
            
        # Global Adaptive Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, 115),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(115, num_classes)
        )

    def forward(self, x):
        # Input shape: [B, 1, 32, 33]
        x = self.stem(x)
        for block in self.blocks:
            res = x
            x = block['rdb'](x)
            x = block['cbam'](x)
            x = block['conv'](x) + res
            
        x = self.pool(x)  # Shape: [B, 104, 1, 1]
        logits = self.classifier(x)  # Shape: [B, 5]
        return logits


# ==========================================
# 4. ResNet18_2D Baseline Model
# ==========================================

class BasicBlock2D(nn.Module):
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_2D(nn.Module):
    """ResNet-18 baseline adapted for 1-channel 2D PSD input [Batch, 1, 32, 33]."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.in_planes = 64

        # 1-channel stem
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock2D(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        logits = self.linear(out)
        return logits


def get_model(model_name="stft_radn", num_classes=5):
    """Factory function to instantiate models."""
    if model_name.lower() in ["stft_radn", "radn"]:
        return STFT_RADN(num_classes=num_classes)
    elif model_name.lower() in ["resnet18", "resnet18_2d"]:
        return ResNet18_2D(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    # Sanity check forward pass and parameter count
    dummy_input = torch.randn(1, 1, 32, 33)
    
    stft_radn = STFT_RADN()
    output = stft_radn(dummy_input)
    param_count = sum(p.numel() for p in stft_radn.parameters() if p.requires_grad)
    print(f"[STFT-RADN] Output shape: {output.shape}, Total trainable params: {param_count:,}")
    assert output.shape == (1, 5), f"Expected shape (1, 5), got {output.shape}"
    
    resnet = ResNet18_2D()
    output_res = resnet(dummy_input)
    param_res = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    print(f"[ResNet18-2D] Output shape: {output_res.shape}, Total trainable params: {param_res:,}")
    assert output_res.shape == (1, 5), f"Expected shape (1, 5), got {output_res.shape}"
    print("[SUCCESS] All model architectures pass forward pass & shape assertions.")
