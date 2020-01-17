import torch
import torch.nn as nn
from spike_layers import SpikeReLU,SpikeConv2d,SpikeLinear,SpikeAvgPool2d,spike_avg_pooling
from spike_tensor import SpikeTensor
import datasets
import argparse

class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # SpikeConv2d can be Conv2d in
        self.conv1 = SpikeConv2d(3, 8, 5, bias=False)
        self.conv2 = SpikeConv2d(8, 16, 3, bias=False)
        self.conv3 = SpikeConv2d(16, 32, 3, bias=False)
        self.fc1 = SpikeLinear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = spike_avg_pooling(out, 2)
        out = self.conv2(out)
        out = spike_avg_pooling(out, 2)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

if __name__=='__main__':
    # parse args
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_file',type=str,help='the location of the trained weights')
    parser.add_argument('--dataset',type=str,default='CIFAR10',help='the location of the trained weights')
    parser.add_argument('--save_file',type=str,help='the output location of the transferred weights')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--timesteps', default=64, type=int)
    parser.add_argument('--reset_mode', default='subtraction', type=str, choices=['zero', 'subtraction'])
    parser.add_argument('--weight_bits', default=4, type=int,help='weight quantization bitwidth')
    args=parser.parse_args()

    # Build Model
    net=ExampleNet()
    if args.load_file:
        net.load_state_dict(torch.load(args.load_file))

    # Preparing for transform
    trainloader,testloader=datasets.get_dataset(args)

    # Weight quantization

