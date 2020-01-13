import torch
import torch.nn as nn
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
        out=self.fc2(out)
        return out

class TestNet5(nn.Module):
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

        self.conv1 = SpikeConv2d(nunits_input, int(16 * scale), 3,padding=1)
        self.conv2 = SpikeConv2d(int(16 * scale), int(32 * scale), 3,padding=1)
        self.conv3 = SpikeConv2d(int(32 * scale), int(32 * scale), 3,padding=1)
        self.conv4 = SpikeConv2d(int(32 * scale), int(32 * scale), 3,padding=1)
        self.conv5 = SpikeConv2d(int(32 * scale), int(32 * scale), 3,padding=1)
        self.conv6 = SpikeConv2d(int(32 * scale), int(32 * scale), 3,padding=1)
        self.pool=SpikeAvgPool2d(2)
        self.fc1 = SpikeLinear(nuintis_fc, outputs)

    def prediction_layer(self,x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        return out

    def forward(self,x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.pool(out)
        out = self.conv6(out)
        out=self.prediction_layer(out)
        return out

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


class TestNet1(nn.Module):
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

        self.conv1 = SpikeConv2d(nunits_input, int(16 * scale), 5,bias=False)
        self.conv2 = SpikeConv2d(int(16 * scale), int(32 * scale), 3,bias=False)
        self.conv3 = SpikeConv2d(int(32 * scale), int(32 * scale), 3,bias=False)
        self.fc1 = SpikeLinear(nuintis_fc, outputs,bias=False)

    def forward(self,x):
        out = self.conv1(x)
        out = spike_avg_pooling(out, 2)
        out = self.conv2(out)
        out = spike_avg_pooling(out, 2)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class TestNet6(nn.Module):
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

        self.conv1 = SpikeConv2d(nunits_input, int(8 * scale), 5,bias=False)
        self.conv2 = SpikeConv2d(int(8 * scale), int(32 * scale), 3,bias=False)
        self.conv3 = SpikeConv2d(int(32 * scale), int(32 * scale), 3,bias=False)
        self.fc1 = SpikeLinear(nuintis_fc, outputs,bias=False)

    def forward(self,x):
        out = self.conv1(x)
        out = spike_avg_pooling(out, 2)
        out = self.conv2(out)
        out = spike_avg_pooling(out, 2)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class TestNet6half(nn.Module):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        if dataset == "CIFAR10":
            nunits_input = 3
            outputs = 10
            nuintis_fc = int(32 * scale) * 4*4
        elif dataset == "ImageNet":
            nunits_input = 3
            outputs = 1000
            nuintis_fc = int(16 * scale) * 52 * 52
        else:
            raise NotImplementedError

        self.conv1 = SpikeConv2d(nunits_input, int(4 * scale), 5,bias=False)
        self.conv2 = SpikeConv2d(int(4 * scale), int(16 * scale), 3,bias=False)
        self.conv3 = SpikeConv2d(int(16 * scale), int(16 * scale), 3,bias=False)
        self.fc1 = SpikeLinear(nuintis_fc, outputs,bias=False)

    def forward(self,x):
        out = self.conv1(x)
        out = spike_avg_pooling(out, 2)
        out = self.conv2(out)
        out = spike_avg_pooling(out, 2)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class TestNet3(nn.Module):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        if dataset == "CIFAR10":
            nunits_input = 3
            outputs = 10
            nuintis_fc = int(32 * scale) * 5*5
        else:
            raise NotImplementedError

        self.conv1 = SpikeConv2d(nunits_input, int(8 * scale), 5,bias=False)
        self.conv2 = SpikeConv2d(int(8 * scale), int(32 * scale), 5,bias=False)
        self.fc1 = SpikeLinear(nuintis_fc, outputs,bias=False)

    def forward(self,x):
        out = self.conv1(x)
        out = spike_avg_pooling(out, 2)
        out = self.conv2(out)
        out = spike_avg_pooling(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

class TestResNet1(nn.Module):
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

        self.conv1 = SpikeConv2d(nunits_input, int(16 * scale), 5,bias=False)
        self.conv2 = SpikeConv2d(int(16 * scale), int(32 * scale), 3,bias=False)
        self.conv3 = SpikeConv2d(int(32 * scale), int(32 * scale), 3,padding=1,bias=False)
        self.fc1 = SpikeLinear(288, outputs)

    def forward(self,x):
        out = self.conv1(x)
        out = spike_avg_pooling(out, 2)
        out = self.conv2(out)
        out = spike_avg_pooling(out, 2)
        conv3_in=out
        out = self.conv3(out)
        out+=conv3_in
        out=spike_avg_pooling(out,2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

if __name__=='__main__':
    net=TestResNet1()
    x=torch.ones([1,3,32,32])
    net(x)
