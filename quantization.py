import torch
from spike_layers import SpikeConv2d
import copy

def quantize_tensor(tensor,bitwidth,channel_level=False):
    if channel_level:
        _max = tensor.abs().view(tensor.size(0),-1).max(1)[0]
    else:
        _max = tensor.abs().max()
    scale = (2 ** (bitwidth - 1) - 1) / _max
    if tensor.dim() == 4:
        scale = scale.view(-1, 1, 1, 1)
    else:
        scale = scale.view(-1, 1)
    new_tensor = torch.round(scale * tensor)
    return new_tensor,scale


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

def quantize_layers(bitwidth,rescale=True):
    for layer in quantized_layers:
        with torch.no_grad():
            # layer.weight.weight_back=layer.weight.weight_back.clamp(-1,1)
            # mean=layer.weight.weight_back.abs().view(layer.weight.size(0),-1).mean(-1).view(-1,*[1 for i in range(layer.weight.dim()-1)])
            quantized_w,scale=quantize_tensor(layer.weight.weight_back,bitwidth,False)
            layer.weight[...]=quantized_w/scale if rescale else quantized_w

def clear_quantize_bindings():
    quantized_layers.clear()

class QuantSGD(torch.optim.SGD):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if hasattr(p,'weight_back'):
                    weight_data=p.weight_back
                else:
                    weight_data=p.data

                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    # TODO: Explore the weight_decay
                    d_p.add_(weight_decay, weight_data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                weight_data.add_(-group['lr'], d_p)
        return loss

def quantize_train(epoch, weight_bitwidth, net, trainloader, optimizer, criterion, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        quantize_layers(weight_bitwidth)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx%60==59:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def quantize_finetune(raw_net,trainloader,criterion,device,args):
    net=copy.deepcopy(raw_net)
    optimizer = QuantSGD(net.parameters(), args.finetune_lr,
                         args.finetune_momentum, weight_decay=args.finetune_wd)
    step_epochs=[int(args.finetune_epochs * 0.5), int(args.finetune_epochs * 0.75)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,step_epochs, 0.1)
    init_quantize_net(net)
    for epoch in range(0, args.finetune_epochs):
        quantize_train(epoch,args.weight_bitwidth,net,trainloader,optimizer,criterion,device)
        lr_scheduler.step(epoch)
    quantize_layers(args.weight_bitwidth,False)
    return net
