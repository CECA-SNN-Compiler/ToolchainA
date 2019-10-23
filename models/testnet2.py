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
        self.fc1=SpikeLinear(3*32*32,outputs)

    def forward(self, x):
        x_=x.view(x.size(0),-1)
        out = self.fc1(x_)
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

        self.fc1 = SpikeLinear(3 * 32 * 32, outputs)

    def forward(self, x):
        x_ = x.view(x.size(0), -1)
        out = self.fc1(x_)
        if out.is_spike:
            out=out.to_float()
        else:
            out=out.data
        out = F.relu(out)
        return out