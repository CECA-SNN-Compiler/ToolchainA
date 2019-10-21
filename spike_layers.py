import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from spike_tensor import *
import numpy as np
from torch.nn.modules.utils import _pair

debug_compare=True

class SpikeModule(nn.Module):
    def set_spike_mode(self):
        for m in self.modules():
            if hasattr(m,'scale_weights'):
                m.scale_weights()

    def set_reset_mode(self,mode):
        for m in self.modules():
            if hasattr(m,'reset_mode'):
                m.reset_mode=mode


class BaseSpikeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.spike_mode = False

class SpikeConv2d(BaseSpikeLayer):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',process_input_im=False,reset_mode='zero',fuse_bn=None):
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
        self.process_input_im=process_input_im
        self.reset_mode=reset_mode

    def forward(self,x:SpikeTensor):
        if self.spike_mode:
            out = F.conv2d(x.data, self.weight_scaled, self.bias_scaled, self.stride, self.padding, self.dilation, self.groups)
            chw=out.size()[1:]
            out_s=out.view(-1,x.timesteps,*chw)
            memb_potential=torch.zeros(out_s.size(0),*chw).to(out_s.device)
            spikes=[]
            for t in range(x.timesteps):
                memb_potential+=self.Vthr*out_s[:,t]
                spike=(memb_potential>self.Vthr).float()
                if self.reset_mode=='zero':
                    memb_potential*=(1- spike)
                elif self.reset_mode=='subtraction':
                    memb_potential-=spike*self.Vthr
                else:
                    raise NotImplementedError
                spikes.append(spike)


            out=SpikeTensor(torch.cat(spikes,0),x.timesteps,self.out_scale_factor)
            if debug_compare:
                float_out=F.conv2d(x.data[:out_s.size(0)], self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                spike_out=out.to_float()
                diff=float_out-spike_out

                2==2
        else:
            out = F.conv2d(x.data, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            self.activations_pool.append(out)
            self.inputs_pool.append(x.data)
            out = SpikeTensor(out, x.timesteps,is_spike=False)
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
        if not self.process_input_im:
            self.weight_scaled.data= self.weight*in_scale.view(1, -1, 1, 1)/out_scale.view(-1,1,1,1)
        else:
            self.weight_scaled.data= self.weight/out_scale.view(-1,1,1,1)
        if self.bias is not None:
            self.bias_scaled.data=self.bias/out_scale
        self.spike_mode=True
        self.activations_pool.clear()
        self.inputs_pool.clear()


class SpikeBatchNorm2d(BaseSpikeLayer):
    def __init__(self):
        super().__init__()