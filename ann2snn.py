import torch
import torch.nn as nn
import torch.nn.functional as F
from spike_layers import *
import copy
from tqdm import tqdm

def warp_spike_layer(layer):
    layer.output_pool=[]
    layer.input_pool=[]
    layer.old_forward=layer.forward
    def forward(x):
        layer.input_pool.append(x.detach().cpu())
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

def trans_layer(layer,warp_layer,device,first_layer):
    assert len(warp_layer.input_pool)!=0 and len(warp_layer.output_pool)!=0
    # get the channel size
    _,inc=warp_layer.input_pool[0].size()[:2]
    _,outc=warp_layer.output_pool[0].size()[:2]
    # cat the activations pool, then get the max 99% number to set the scale
    inputs=torch.cat(warp_layer.input_pool,0).transpose(0, 1).contiguous().view(inc, -1)
    outputs=torch.cat(warp_layer.output_pool,0).transpose(0, 1).contiguous().view(outc, -1)
    out_scale = torch.sort(outputs, -1)[0][:, int(outputs.size(1) * 0.99)]
    in_scale = torch.sort(inputs, -1)[0][:, int(inputs.size(1) * 0.99)]
    # free the memory
    warp_layer.input_pool.clear()
    warp_layer.output_pool.clear()
    # set the scale dimension to rescale the weights of original layer
    spatial_dim=layer.weight.dim()-2
    in_scale=in_scale.view(1,-1,*[1]*spatial_dim).to(device)
    out_scale=out_scale.view(-1,1,*[1]*spatial_dim).to(device)
    # if it is the first layer then close the in scale
    if first_layer:
        in_scale=1
    # scale the weights
    layer.weight.data=layer.weight.data*in_scale/out_scale
    if layer.bias is not None:
        layer.bias.data=layer.bias.data/out_scale.view(-1)
    # set the out_scales of layer
    layer.out_scales=out_scale.view(-1)

def trans_ann2snn(ann,dataloader,device):
    print("transfer ann to snn")

    # switch to evaluate mode
    ann.eval()
    trans_cnt=0

    for layer in ann.modules():
        if isinstance(layer,SpikeConv2d) or isinstance(layer,SpikeLinear):
            warp_spike_layer(layer)
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            output = ann(data)


    for layer in ann.modules():
        if isinstance(layer,SpikeConv2d) or isinstance(layer,SpikeLinear):
            trans_layer(layer, layer, device, first_layer=trans_cnt == 0)
            unwarp_spike_layer(layer)
            trans_cnt+=1
    return ann

