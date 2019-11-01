import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *
from utils import fuse_conv_bn_eval


class TestNetFake(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(4*4*32, 10)

    def forward(self, x):
        out=self.conv1(x)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class TestNetOriginal(nn.Module):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        if dataset == "CIFAR10":
            outputs = 10
            nuintis_fc = int(32 * scale) * 4*4
        elif dataset == "ImageNet":
            outputs = 1000
            nuintis_fc = int(32 * scale) * 52 * 52
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
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

        self.conv1 = SpikeConv2d(nunits_input, int(16 * scale), 5,first_layer=True)
        self.bn1 = nn.BatchNorm2d(int(16 * scale))
        self.conv2 = SpikeConv2d(int(16 * scale), int(32 * scale), 3)
        # self.conv2 = nn.Conv2d(int(16 * scale), int(32 * scale), 3)
        self.bn2 = nn.BatchNorm2d(int(32 * scale))
        self.conv3 = SpikeConv2d(int(32 * scale), int(32 * scale), 3)
        self.bn3 = nn.BatchNorm2d(int(32 * scale))
        self.fc1 = SpikeLinear(nuintis_fc, outputs)
    
    def fuse_conv_bn(self):
        self.eval()
        self.conv1,self.bn1 = fuse_conv_bn_eval(self.conv1, self.bn1)
        self.conv2,self.bn2 = fuse_conv_bn_eval(self.conv2, self.bn2)
        self.conv3,self.bn3 = fuse_conv_bn_eval(self.conv3, self.bn3)

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
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.bn2(self.conv2(out)))
            out = F.max_pool2d(out, 2)
            out = F.relu(self.bn3(self.conv3(out)))
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
        return out