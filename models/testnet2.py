import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *


class TestNet2(nn.Module):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        outputs=10
        self.fc1=SpikeLinear(3*32*32,32)
        self.fc2=SpikeLinear(32,outputs)

    def forward(self, x):
        x_=x.view(x.size(0),-1)
        out = self.fc1(x_)
        out=F.relu(out)
        out=self.fc2(out)
        return out
