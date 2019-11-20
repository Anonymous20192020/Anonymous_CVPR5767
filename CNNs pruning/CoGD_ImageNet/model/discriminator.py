import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, num_classes=1000):
        super(Discriminator, self).__init__()
        self.filters = [num_classes, 512, 256, 1]

        block = [

            nn.Linear(self.filters[i], self.filters[i+1]) \
            for i in range(3)
        ]
        self.body = nn.Sequential(*block)

        self._initialize_weights()

    def forward(self, input):
        x = self.body(input)
        # print(x.shape,'dddddddddddddddddddd')
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Discriminator_mmd(nn.Module):
    def __init__(self):
        super(Discriminator_mmd, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod((320, 4, 4))), 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
