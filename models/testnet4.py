import torch
import torch.nn as nn
from spike_layers import *


class TestNet4(nn.Module):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        if dataset == "CIFAR10":
            nunits_input = 3
            outputs = 10
            nuintis_fc = int(32 * scale) * 4*4
        elif dataset == "ImageNet":
            nunits_input = 3
            outputs = 1000
            nuintis_fc = int(32 * scale) * 52 * 52
        else:
            raise NotImplementedError

        self.conv1 = SpikeConv2d(nunits_input, int(16 * scale), 5)
        self.conv2 = SpikeConv2d(int(16 * scale), int(32 * scale), 3)
        self.conv3 = SpikeConv2d(int(32 * scale), int(32 * scale), 3)
        self.fc1 = SpikeLinear(nuintis_fc, outputs)

    def forward(self,x):
        out = self.conv1(x)
        out = spike_avg_pooling(out, 2)
        out = self.conv2(out)
        out = spike_avg_pooling(out, 2)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out