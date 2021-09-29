import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, channel_in, channel_out, dropout=0.0):
        super().__init__()

        self.module1 = nn.Sequential(nn.Conv2d(channel_in, 128, 7, padding=3),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.module2 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1, stride=2),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.across2 = nn.Conv2d(128, 256, 1, stride=2)
        self.module21 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.module22 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.module3 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1, stride=2),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(512, channel_out, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.across3 = nn.Conv2d(256, 512, 1, stride=2)
        self.module31 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(512, channel_out, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.module32 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(512, channel_out, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        self.module33 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Conv2d(512, channel_out, 3, padding=1),
                                nn.ReLU(),
                                nn.Dropout(dropout))

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x) + self.across2(x)
        x = self.module21(x) + x
        x = self.module22(x) + x
        x = self.module3(x) + self.across3(x)
        x = self.module31(x) + x
        x = self.module32(x) + x
        x = self.module33(x) + x
        return x
