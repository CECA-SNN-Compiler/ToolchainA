import torch
import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *
import copy
from tqdm import tqdm

def warp_spike_layer(layer):
    layer.output_pool=[]
    layer.old_forward=layer.forward
    def forward(x):
        out=layer.old_forward(x)
        layer.output_pool.append(out.detach().cpu())
        return out
    layer.forward=forward
    return layer

def unwarp_spike_layer(layer):
    layer.output_pool = None
    layer.input_pool = None
    layer.forward=layer.old_forward
    return layer

def trans_layer(layer,prev_layer,device):
    assert len(layer.output_pool)!=0
    spatial_dim = layer.weight.dim() - 2
    if prev_layer is None:
        in_scale=1
    else:
        _, inc = prev_layer.output_pool[0].size()[:2]
        inputs = torch.cat(prev_layer.output_pool, 0).transpose(0, 1).contiguous().view(inc, -1)
        in_scale = torch.sort(inputs, -1)[0][:, int(inputs.size(1) * 0.99)]
        if in_scale.size(0)!=layer.weight.size(1):
            if isinstance(layer,SpikeLinear) and isinstance(prev_layer,SpikeConv2d):
                repeat=int(layer.weight.size(1)/in_scale.size(0))
                in_scale=in_scale.view(-1,1).repeat_interleave(repeat,1).view(-1)
                2==2
        in_scale = in_scale.view(1, -1, *[1] * spatial_dim).to(device)
    _, outc = layer.output_pool[0].size()[:2]
    outputs=torch.cat(layer.output_pool,0).transpose(0, 1).contiguous().view(outc, -1)
    out_scale = torch.sort(outputs, -1)[0][:, int(outputs.size(1) * 0.99)]
    out_scale = out_scale.view(-1, 1, *[1] * spatial_dim).to(device)

    # set the scale dimension to rescale the weights of original layer

    # scale the weights
    layer.weight.data=layer.weight.data*in_scale/out_scale
    if layer.bias is not None:
        layer.bias.data=layer.bias.data/out_scale.view(-1)
    # set the out_scales of layer
    layer.out_scales=out_scale.view(-1)

def trans_ann2snn(ann,dataloader,device):
    print("Start transfer ANN to SNN, this will take a while")

    # switch to evaluate mode
    ann.eval()

    for layer in ann.modules():
        if isinstance(layer,SpikeConv2d) or isinstance(layer,SpikeLinear):
            warp_spike_layer(layer)
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            output = ann(data)

    prev_layer=None
    for layer in ann.modules():
        if isinstance(layer,SpikeConv2d) or isinstance(layer,SpikeLinear):
            trans_layer(layer, prev_layer, device)
            prev_layer=layer
    for layer in ann.modules():
        if isinstance(layer, SpikeConv2d) or isinstance(layer, SpikeLinear):
            unwarp_spike_layer(layer)
    print("Transfer ANN to SNN Finished")
    return ann

