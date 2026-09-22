"""
A small CIFAR-style ResNet (the "ResNet-20/32/44/56" family from the original
He et al. 2015 paper, section 4.2 — designed specifically for 32x32 images,
not the ImageNet ResNet-18/50 you see in torchvision).

Layout: one 3x3 stem conv, then three stages of BasicBlocks at 16 -> 32 -> 64
channels, then global average pool -> linear classifier. Each stage's first
block halves the spatial resolution (stride 2) except the very first stage.

All weights are randomly initialized by PyTorch's default nn.Conv2d/nn.Linear
init (Kaiming-uniform) -- nothing here loads a pretrained state_dict.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    TODO (this is yours to write): the residual block.

    A BasicBlock does:
        out = conv3x3(x)  -> bn -> relu -> conv3x3 -> bn
        out = out + shortcut(x)
        out = relu(out)

    where `shortcut(x)` is:
        - the identity (just `x`) when in_channels == out_channels and stride == 1
        - a projection (1x1 conv + bn, stride=`stride`) when the shape changes,
          e.g. at the start of a new stage where channels double and stride=2

    Why the skip connection at all? Without it, stacking many conv layers makes
    gradients have to flow through every conv/bn/relu in the chain during
    backprop, and in practice deep plain CNNs get *harder* to optimize past a
    certain depth (not just overfitting -- training error itself gets worse).
    The `+ shortcut(x)` gives gradients a direct path back, and gives each
    block the easy option of learning "do nothing" (identity) if that's
    already good enough, rather than having to learn an identity mapping
    through a stack of nonlinear layers.

    self.conv1, self.bn1, self.conv2, self.bn2 are already declared below --
    use them in forward(). You need to also declare `self.shortcut` in
    __init__ (nn.Identity() or a Sequential(conv1x1, bn) depending on whether
    shapes change) and write forward().
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # TODO: build self.shortcut here.
        # Hint: shapes change (need a projection) exactly when stride != 1
        # or in_channels != out_channels.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass described in the docstring above.
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet-(6n+2) for CIFAR: n BasicBlocks per stage, 3 stages, 16/32/64 channels.
    n=3 -> ResNet-20 (a good speed/accuracy default for a 6GB GPU).
    """

    def __init__(self, num_blocks_per_stage: int = 3, num_classes: int = 10):
        super().__init__()
        self.in_channels = 16

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(16, num_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(32, num_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(64, num_blocks_per_stage, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
