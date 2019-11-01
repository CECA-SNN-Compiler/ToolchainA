import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from spike_tensor import *
import numpy as np
from torch.nn.modules.utils import _pair
from torch.nn import init
import math


class LayerManager():
    def __init__(self):
        self.debug_compare=False

manager=LayerManager()

class SpikeModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.spike_mode=False

    def set_spike_mode(self):
        for m in self.modules():
            if hasattr(m,'scale_weights'):
                m.scale_weights()
                m.spike_mode = True
        self.spike_mode=True

    def set_reset_mode(self,mode):
        for m in self.modules():
            if hasattr(m,'reset_mode'):
                m.reset_mode=mode

    def fuse_conv_bn(self):
        return


class BaseSpikeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.spike_mode = False

class SpikeConv2d(BaseSpikeLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', first_layer=False, reset_mode='zero', fuse_bn=None):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight= Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        self.Vthr=1
        self.weight_scaled=Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias_scaled = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_scaled', None)
        self.activations_pool=[]
        self.inputs_pool=[]
        self.out_scale_factor=Parameter(torch.Tensor(out_channels))
        self.first_layer=first_layer
        self.reset_mode=reset_mode

    def forward(self,x):
        if self.spike_mode:
            assert isinstance(x,SpikeTensor)
            out = F.conv2d(x.data, self.weight_scaled, self.bias_scaled, self.stride, self.padding, self.dilation, self.groups)
            chw=out.size()[1:]
            out_s=out.view(x.timesteps,-1,*chw)
            memb_potential=torch.zeros(out_s.size(1),*chw).to(out_s.device)
            spikes=[]
            for t in range(x.timesteps):
                memb_potential+=self.Vthr*out_s[t]
                spike=(memb_potential>self.Vthr).float()
                if self.reset_mode=='zero':
                    memb_potential*=(1- spike)
                elif self.reset_mode=='subtraction':
                    memb_potential-=spike*self.Vthr
                else:
                    raise NotImplementedError
                spikes.append(spike)

            out=SpikeTensor(torch.cat(spikes,0),x.timesteps,self.out_scale_factor)
            if manager.debug_compare:
                float_out=F.conv2d(x.to_float(), self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                float_out=F.relu(float_out)
                spike_out=out.to_float()
                diff=float_out-spike_out
                print('diff', diff.mean(0))
                print('float', float_out.mean(0))
                print('spike', spike_out.mean(0))
                print('Frac', diff.mean(0) / (spike_out.mean(0)+1e-8))
        else:
            out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            self.activations_pool.append(F.relu(out))
            self.inputs_pool.append(x)
        return out

    def scale_weights(self):
        """
        $ W^l := W^l * \lambda^{l-1} \lambda^l $ and $ b^l := b^l / \lambda^l $
        """
        activations = torch.cat(self.activations_pool, 0).transpose(0, 1).contiguous().view(self.out_channels, -1)
        inputs = torch.cat(self.inputs_pool, 0).transpose(0, 1).contiguous().view(self.in_channels, -1)
        out_scale=torch.sort(activations,-1)[0][:,int(activations.size(1)*0.99)]
        in_scale=torch.sort(inputs,-1)[0][:,int(inputs.size(1)*0.99)]
        self.out_scale_factor.data=out_scale
        if not self.first_layer:
            self.weight_scaled.data= self.weight*in_scale.view(1, -1, 1, 1)/out_scale.view(-1,1,1,1)
        else:
            self.weight_scaled.data= self.weight/out_scale.view(-1,1,1,1)
            self.weight.data= self.weight/out_scale.mean()
        if self.bias is not None:
            self.bias_scaled.data=self.bias/out_scale
            self.bias.data=self.bias/out_scale.mean()
        self.activations_pool.clear()
        self.inputs_pool.clear()


class SpikeLinear(BaseSpikeLayer):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self,in_features, out_features, bias=True,reset_mode='zero',first_layer=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight_scaled = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.bias_scaled = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.Vthr=1
        self.reset_parameters()
        self.reset_mode=reset_mode
        self.activations_pool = []
        self.inputs_pool = []
        self.out_scale_factor = Parameter(torch.Tensor(out_features))
        self.first_layer=first_layer

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.spike_mode:
            out = F.linear(x.data, self.weight_scaled, self.bias_scaled)
            chw=out.size()[1:]
            out_s=out.view(x.timesteps,-1,*chw)
            memb_potential=torch.zeros(out_s.size(1),*chw).to(out_s.device)
            spikes=[]
            for t in range(x.timesteps):
                memb_potential+=self.Vthr*out_s[t]
                spike=(memb_potential>self.Vthr).float()
                if self.reset_mode=='zero':
                    memb_potential*=(1- spike)
                elif self.reset_mode=='subtraction':
                    memb_potential-=spike*self.Vthr
                else:
                    raise NotImplementedError
                spikes.append(spike)

            out=SpikeTensor(torch.cat(spikes,0),x.timesteps,self.out_scale_factor)
            if manager.debug_compare:
                float_out=F.relu(F.linear(x.to_float(), self.weight, self.bias))
                spike_out=out.to_float()
                diff=float_out-spike_out
                print('diff',diff.mean(0))
                print('float',float_out.mean(0))
                print('spike',spike_out.mean(0))
                print('Frac',diff.mean(0)/spike_out.mean(0))
        else:
            out = F.linear(x, self.weight, self.bias)
            self.activations_pool.append(out)
            self.inputs_pool.append(F.relu(x.data))
        return out

    def scale_weights(self):
        """
        $ W^l := W^l * \lambda^{l-1} \lambda^l $ and $ b^l := b^l / \lambda^l $
        """
        activations = torch.cat(self.activations_pool, 0).transpose(0, 1).contiguous().view(self.out_features, -1)
        inputs = torch.cat(self.inputs_pool, 0).transpose(0, 1).contiguous().view(self.in_features, -1)
        out_scale = torch.sort(activations, -1)[0][:, int(activations.size(1) * 0.999)]+1e-8
        in_scale = torch.sort(inputs, -1)[0][:, int(inputs.size(1) * 0.999)]+1e-8
        self.out_scale_factor.data = out_scale
        if not self.first_layer:
            self.weight_scaled.data = self.weight * in_scale.view(1, -1) / out_scale.view(-1, 1)
        else:
            self.weight_scaled.data = self.weight / out_scale.view(-1, 1)
            # self.weight.data = self.weight / out_scale.mean()
        if self.bias is not None:
            self.bias_scaled.data = self.bias / out_scale
            # self.bias.data = self.bias / out_scale.mean()
        self.activations_pool.clear()
        self.inputs_pool.clear()

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def spike_pooling(x,kernel_size, stride=None, padding=0,mode='max'):
    assert isinstance(x,SpikeTensor)
    if mode=='max':
        # TODO: Optimize
        firing_ratio=x.firing_ratio()
        _,ind=F.max_pool2d(firing_ratio,kernel_size,stride,padding,return_indices=True)
        outh,outw=ind.size()[-2:]
        _x=x.timestep_dim_tensor()
        t,b,c,h,w=_x.size()
        _x=_x.view(t,b,c,-1)
        ind=ind.view(1,b,c,-1).repeat_interleave(t,0)
        out=_x.gather(-1,ind)
        out=out.view(b*t,c,outh,outw)
    elif mode=='avg':
        # TODO: Real AVG POOLING
        out = F.avg_pool2d(x.data, kernel_size, stride, padding)
        raise NotImplementedError
    else:
        raise NotImplementedError
    return SpikeTensor(out,x.timesteps,scale_factor=x.scale_factor)