# model.py

import torch
import torch.nn as nn
import torchvision.models as models

def conv_block(in_ch, out_ch, ks=3, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ks, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )

class ImprovedCNN(nn.Module):
    """
    Improved CNN:
    - Conv-BN-ReLU blocks
    - Progressive channel growth (16, 32, 64)
    - Dropout before final classifier
    - AdaptiveAvgPool2d to make model robust to spatial dims
    - Returns logits (use BCEWithLogitsLoss for binary)
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 16),   # e.g. 28x28 -> 14x14
            conv_block(16, 32),            # 14x14 -> 7x7
            conv_block(32, 64),            # 7x7 -> 3x3 (floor)
            nn.AdaptiveAvgPool2d((1, 1))   # -> (N, 64, 1, 1)
        )
        out_dim = 64
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, 1 if num_classes == 2 else num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits

class RobustResNet(nn.Module):
    """
    ResNet18-based model adapted for single-channel medical images and binary classification.
    - Use use_pretrained=True to load imagenet weights (recommended if internet/weights available).
    - Adjusts first conv to accept in_channels (e.g. 1).
    - Returns logits (use BCEWithLogitsLoss for training).
    """
    def __init__(self, in_channels=1, num_classes=2, use_pretrained=False):
        super().__init__()
        self.num_classes = num_classes

        # Load base ResNet18
        self.backbone = models.resnet18(pretrained=use_pretrained)

        # Replace first conv to accept in_channels (preserve approx. statistics if pretrained)
        if in_channels != 3:
            old_conv = self.backbone.conv1
            new_conv = nn.Conv2d(in_channels,
                                 old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=(old_conv.bias is not None))
            if use_pretrained:
                # If pretrained, average weights across RGB channels to init single-channel conv
                with torch.no_grad():
                    new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
                    if old_conv.bias is not None:
                        new_conv.bias[:] = old_conv.bias
            self.backbone.conv1 = new_conv

        # Replace final classifier: for binary return single logit
        if num_classes == 2:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        else:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # small head improvements
        self.dropout = nn.Dropout(p=0.4)

        # initialize new fc if not pretrained
        if not use_pretrained:
            nn.init.xavier_uniform_(self.backbone.fc.weight)
            if self.backbone.fc.bias is not None:
                nn.init.zeros_(self.backbone.fc.bias)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.fc(x)
        return x  # logits

# --- Bagian untuk pengujian singkat ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1

    print("--- Menguji Model 'ImprovedCNN' ---")
    model = ImprovedCNN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)

    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)

    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'ImprovedCNN' berhasil.")

    print("\n--- Menguji Model 'RobustResNet' ---")
    model = RobustResNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, use_pretrained=False)
    print(model)
    dummy = torch.randn(8, IN_CHANNELS, 224, 224)  # ResNet expects ~224x224; resize your dataset or adapt transforms
    out = model(dummy)
    print("Output shape:", out.shape)