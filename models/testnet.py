import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *


class TestNetOriginal(nn.Module):
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

        self.conv1 = nn.Conv2d(nunits_input, int(16 * scale), 5)
        self.bn1 = nn.BatchNorm2d(int(16 * scale))
        self.conv2 = nn.Conv2d(int(16 * scale), int(32 * scale), 3)
        self.bn2 = nn.BatchNorm2d(int(32 * scale))
        self.conv3 = nn.Conv2d(int(32 * scale), int(32 * scale), 3)
        self.bn3 = nn.BatchNorm2d(int(32 * scale))
        self.fc1 = nn.Linear(nuintis_fc, outputs)

    def forward(self, x):
        conv1_out=self.conv1(x)
        out = F.relu(self.bn1(conv1_out))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class TestNet(SpikeModule):
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
        self.bn1 = nn.BatchNorm2d(int(16 * scale))
        self.conv2 = nn.Conv2d(int(16 * scale), int(32 * scale), 3)
        self.bn2 = nn.BatchNorm2d(int(32 * scale))
        self.conv3 = nn.Conv2d(int(32 * scale), int(32 * scale), 3)
        self.bn3 = nn.BatchNorm2d(int(32 * scale))
        self.fc1 = nn.Linear(nuintis_fc, outputs)

    def forward(self, x):
        conv1_out=self.conv1(x)
        if conv1_out.is_spike:
            x=conv1_out.to_float()
        else:
            x=conv1_out.data
        out = F.relu(self.bn1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out