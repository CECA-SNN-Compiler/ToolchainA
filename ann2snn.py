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

def trans_layer(layer,prev_layer,timesteps,weight_bits):
    assert len(layer.output_pool)!=0
    out_max=torch.cat(layer.output_pool,0).max()
    weight_max=layer.weight.abs().max()
    S_w=layer.weight.data.abs().max()/(2**(weight_bits-1)-1)
    if prev_layer is None:
        Vthr = torch.round(out_max / S_w )
    else:
        _, inc = prev_layer.output_pool[0].size()[:2]
        in_max = torch.cat(prev_layer.output_pool, 0).max()
        S_in = in_max / timesteps
        Vthr=torch.round(out_max/S_in/S_w/timesteps)
    W=torch.round((2**(weight_bits-1)-1)/weight_max*layer.weight.data)
    layer.weight.data[...]=W
    layer.Vthr[...]=Vthr
    layer.out_scales[...]=1/Vthr/timesteps

    if layer.bias is not None:
        raise NotImplementedError
        # layer.bias.data=layer.bias.data/out_scale.view(-1)
    # set the out_scales of layer

    # print scale
    print("Layer Mean:",torch.mean(layer.weight.data).item(),
          "STD:",torch.std(layer.weight.data).item(),
          "Vthr",layer.Vthr.item())


def trans_ann2snn(raw_net, dataloader, device, timesteps, weight_bits):
    print("Start transfer ANN to SNN, this will take a while")

    # switch to evaluate mode
    net=copy.deepcopy(raw_net)
    net.eval()

    for layer in net.modules():
        if isinstance(layer,SpikeConv2d) or isinstance(layer,SpikeLinear):
            warp_spike_layer(layer)
    with torch.no_grad():
        for data, target in tqdm(dataloader):
            data = data.to(device)
            output = net(data)

    prev_layer=None
    for layer in net.modules():
        if isinstance(layer,SpikeConv2d) or isinstance(layer,SpikeLinear):
            trans_layer(layer, prev_layer, timesteps,weight_bits)
            prev_layer=layer
    for layer in net.modules():
        if isinstance(layer, SpikeConv2d) or isinstance(layer, SpikeLinear):
            unwarp_spike_layer(layer)
    print(f"Transfer ANN to SNN (timesteps={timesteps}, weight_bits={weight_bits}) Finished")
    return net
