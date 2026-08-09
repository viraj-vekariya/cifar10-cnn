import torch.nn as nn


class ConvBlock(nn.Module):
    """Two conv-bn-relu layers followed by a 2x2 maxpool.

    BatchNorm is placed between the conv and the activation (conv -> bn -> relu),
    not after it. BN normalizes the pre-activation distribution so ReLU sees a
    consistent input scale across training, which is the original Ioffe/Szegedy
    ordering and the one that empirically trains faster/more stably here.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class CIFAR10CNN(nn.Module):
    """VGG-style CNN for CIFAR-10: 3 conv blocks (6 conv layers total), then a
    2-layer classifier head.

    Design choices:
    - 3 blocks doubling channels (64 -> 128 -> 256) is the standard pattern for
      keeping compute roughly balanced as spatial resolution halves each block
      (32x32 -> 16x16 -> 8x8 -> 4x4). Deep enough to need BatchNorm/dropout to
      train well (unlike a shallow 2-conv-layer net), shallow enough to stay
      readable and trainable from scratch on a laptop in well under an hour.
    - Two conv layers per block (rather than one) before each pool gives more
      representational capacity per spatial resolution, which matters more on
      CIFAR-10's small 32x32 images than adding more pooling stages would.
    - Dropout only in the classifier head (not in conv blocks, where BatchNorm
      already provides regularization) — standard split of responsibility:
      BN stabilizes/regularizes conv feature extraction, dropout regularizes
      the high-parameter-count fully-connected layers where overfitting is
      concentrated.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    import torch

    model = CIFAR10CNN()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print("output shape:", out.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")
