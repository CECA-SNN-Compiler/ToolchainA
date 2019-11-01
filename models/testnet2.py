import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *


class TestNet2Original(nn.Module):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        outputs=10
        self.fc1=nn.Linear(3*32*32,32)
        self.fc2=nn.Linear(32,outputs)

    def forward(self, x):
        x_=x.view(x.size(0),-1)
        out = self.fc1(x_)
        out=F.relu(out)
        out=self.fc2(out)
        return out

class TestNet2(SpikeModule):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        outputs = 10
        self.fc1 = SpikeLinear(3 * 32 * 32, 32,first_layer=True)
        self.fc2 = SpikeLinear(32, outputs)

    def forward(self, x):
        x_ = x.view(x.size(0), -1)

        if self.spike_mode:
            out = self.fc1(x_)
            out = self.fc2(out)
            out=out.to_float()
        else:
            out = self.fc1(x_)
            out = F.relu(out)
            out = self.fc2(out)
        return out