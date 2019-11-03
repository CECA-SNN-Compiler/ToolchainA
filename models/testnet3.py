import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *


class TestNet3(SpikeModule):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        self.conv1 = SpikeConv2d(3, 16, 5,bias=False)
        self.conv2 = SpikeConv2d(16, 32, 3,bias=False)
        self.conv3 = SpikeConv2d(32, 32, 3,bias=False)
        self.fc1 = SpikeLinear(4 * 4 * 32, 10)

    def forward(self, x):
        if self.spike_mode:
            out = self.conv1(x)
            out = spike_pooling(out, 2,mode='max')
            out=self.conv2(out)
            out = spike_pooling(out, 2,mode='max')
            out = self.conv3(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = out.to_float()
        else:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.conv3(out))
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
        return out