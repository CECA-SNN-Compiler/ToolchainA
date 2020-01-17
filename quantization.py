import torch
from spike_layers import SpikeConv2d

quantized_layers = []
def init_quantize_net(net):
    for name,m in net.named_modules():
        if isinstance(m,SpikeConv2d):# or isinstance(m,SpikeLinear):
        # if isinstance(m,SpikeConv2d):
            if hasattr(m.weight,'weight_back'):
                continue
            quantized_layers.append(m)
            m.weight.weight_back=m.weight.data.clone()
            if m.bias is not None:
                raise NotImplementedError

def quantize_layers():
    for layer in quantized_layers:
        with torch.no_grad():
            # layer.weight.weight_back=layer.weight.weight_back.clamp(-1,1)
            # mean=layer.weight.weight_back.abs().view(layer.weight.size(0),-1).mean(-1).view(-1,*[1 for i in range(layer.weight.dim()-1)])
            channel_max=layer.weight.weight_back.abs().view(layer.weight.size(0),-1).max(1)[0]
            quant = (2**(args.bitwidth-1)-1) / channel_max
            if layer.weight.dim()==4:
                quant=quant.view(-1,1,1,1)
            else:
                quant = quant.view(-1, 1)
            layer.weight[...]=torch.round(quant*layer.weight.weight_back)/quant
