import torch
import torch.nn as nn
import torchvision.models as models

def conv_block(in_ch, out_ch, ks=3, padding=1, dropout=0.25):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(p=dropout)
    )

class ImprovedCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 32, dropout=0.1),
            conv_block(32, 64, dropout=0.15),
            conv_block(64, 128, dropout=0.25),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1 if num_classes == 2 else num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class RobustResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, use_pretrained=False):
        super().__init__()
        base = models.resnet18(pretrained=use_pretrained)
        if in_channels != 3:
            old_conv = base.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, 
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=False)
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
            base.conv1 = new_conv

        base.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(base.fc.in_features, 1 if num_classes == 2 else num_classes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)
