from spike_layers import SpikeReLU,SpikeConv2d,SpikeLinear,spike_avg_pooling
from spike_tensor import SpikeTensor
import torch.nn as nn

class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # SpikeConv2d can be Conv2d in
        self.conv1 = SpikeConv2d(3, 8, 5, bias=False)
        self.relu1=SpikeReLU()
        self.conv2 = SpikeConv2d(8, 16, 3, bias=False)
        self.relu2 = SpikeReLU()
        self.conv3 = SpikeConv2d(16, 32, 3, bias=False)
        self.relu3 = SpikeReLU()
        self.fc1 = SpikeLinear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = spike_avg_pooling(out, 2)
        out = self.relu2(self.conv2(out))
        out = spike_avg_pooling(out, 2)
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out