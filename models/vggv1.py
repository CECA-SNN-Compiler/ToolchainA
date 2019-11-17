import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *

cfg = {
	'VGG5' : [64, 'A', 128, 'D', 128, 'A'],
	'VGG9': [64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'A', 512, 'D', 512, 'D'],
	'VGG11': [64, 'A', 128, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'D'],
	'VGG16': [64, 'D', 64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'D', 512, 'D']
}


class VGG16V1(nn.Module):
    def __init__(self,dataset="CIFAR10",scale=1):
        super().__init__()
        if dataset=='CIFAR10':
            self.labels=10
        self.features, self.classifier = self._make_layers(cfg['VGG16'])

    def _make_layers(self,cfg):
        layers = []
        in_channels = 3

        for x in (cfg):
            stride = 1

            if x == 'A':
                layers += [SpikeAvgPool2d(kernel_size=2, stride=2)]
            elif x=='D':
                continue
            else:
                layers += [SpikeConv2d(in_channels, x, kernel_size=3, padding=1, stride=stride, bias=False),
                           ]
                in_channels = x

        features = nn.Sequential(*layers)

        layers = []
        layers += [SpikeLinear(512 * 2 * 2, 4096, bias=False)]
        layers += [SpikeLinear(4096, 4096, bias=False)]
        layers += [SpikeLinear(4096, self.labels, bias=False)]

        classifer = nn.Sequential(*layers)
        return features,classifer

    def forward(self, x):
        for layer in self.features:
            x=layer(x)
        bt,c,h,w=x.size()
        x=x.view(bt,-1)
        out=self.classifier(x)
        return out


if __name__=='__main__':
    net=VGG16V1()
    x=torch.ones([1,3,32,32])
    net(x)
