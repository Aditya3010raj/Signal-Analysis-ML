# model.py
import torch.nn as nn

class SpectrumCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B,1,F,T)
        f = self.net(x)            # (B,64,1,1)
        f = f.view(f.size(0), -1)  # (B,64)
        logits = self.classifier(f) # (B, n_classes)
        return logits
