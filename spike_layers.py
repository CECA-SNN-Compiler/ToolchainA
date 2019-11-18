import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.utils import _pair
from spike_tensor import SpikeTensor
import math

reset_mode='subtraction'



class SpikeConv2d(nn.Conv2d):

    def forward(self,x):
        Vthr=1
        if isinstance(x,SpikeTensor):
            out = F.conv2d(x.data, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.groups)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            memb_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = []
            for t in range(x.timesteps):
                memb_potential += Vthr* out_s[t]
                spike = (memb_potential > Vthr).float()
                if reset_mode == 'zero':
                    memb_potential *= (1 - spike)
                elif reset_mode == 'subtraction':
                    memb_potential -= spike * Vthr
                else:
                    raise NotImplementedError
                spikes.append(spike)

            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps,self.out_scales)
            return out
        else:
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            out = F.relu(out)
            return out

class SpikeLinear(nn.Linear):
    def forward(self,x):
        self.Vthr=1
        if isinstance(x,SpikeTensor):
            out = F.linear(x.data, self.weight, self.bias)
            chw = out.size()[1:]
            out_s = out.view(x.timesteps, -1, *chw)
            memb_potential = torch.zeros(out_s.size(1), *chw).to(out_s.device)
            spikes = []
            for t in range(x.timesteps):
                memb_potential += self.Vthr * out_s[t]
                spike = (memb_potential > self.Vthr).float()
                if reset_mode == 'zero':
                    memb_potential *= (1 - spike)
                elif reset_mode == 'subtraction':
                    memb_potential -= spike * self.Vthr
                else:
                    raise NotImplementedError
                spikes.append(spike)

            out = SpikeTensor(torch.cat(spikes, 0), x.timesteps, self.out_scales)
            return out
        else:
            out = F.linear(x, self.weight, self.bias)
            out = F.relu(out)
            return out

class SpikeAvgPool2d(nn.Module):
    def __init__(self,kernel_size, stride=None, padding=0):
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        super().__init__()

    def forward(self,x):
        return spike_avg_pooling(x,self.kernel_size,self.stride,self.padding)


def spike_avg_pooling(x,kernel_size, stride=None, padding=0):
    if isinstance(x, SpikeTensor):
        out=F.avg_pool2d(x.data, kernel_size, stride, padding)
        spike_out=SpikeTensor(out,x.timesteps, x.scale_factor)
        return spike_out
    else:
        out=F.avg_pool2d(x, kernel_size, stride, padding)
        return out
