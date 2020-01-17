from datasets import get_dataset
import argparse
import torch
import numpy as np
import time
import torch.nn as nn
from spike_layers import *
import GPUtil
import os
import copy
import matplotlib.pyplot as plt

min_mem_gpu = np.argmin([_.memoryUsed for _ in GPUtil.getGPUs()])
print("selecting GPU {}".format(min_mem_gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = str(min_mem_gpu)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(test_loader, model, device, criterion, epoch, train_writer=None,spike_mode=False):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if spike_mode==False:
        prediction_layer=None
        for layer in model.modules():
            if isinstance(layer,SpikeLinear):
                prediction_layer=layer

    end = time.time()
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test

            data = data.to(device)
            if spike_mode:
                if args.input_poisson:
                    assert NotImplementedError
                else:
                    replica_data=torch.cat([data for _ in range(args.timesteps)],0)
                    data = SpikeTensor(replica_data, args.timesteps,scale_factor=1)

            output = model(data)
            if isinstance(output,SpikeTensor):
                output=output.to_float()
            # elif hasattr(prediction_layer,'out_scales'):
            #     output=output*prediction_layer.out_scales.view(1,-1)

            target = target.to(device)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}'.format(
        top1=top1, top5=top5, batch_time=batch_time, losses=losses))
    # log to TensorBoard
    if train_writer is not None:
        train_writer.add_scalar('val_loss', losses.avg, epoch)
        train_writer.add_scalar('val_acc', top1.avg, epoch)

    return top1.avg, losses.avg

def error_validate(test_loader, model, device, criterion, epoch, train_writer=None,spike_mode=False):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if spike_mode==False:
        prediction_layer=None
        for layer in model.modules():
            if isinstance(layer,SpikeLinear):
                prediction_layer=layer

    end = time.time()
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test

            raw_data = data.to(device)
            if args.input_poisson:
                assert NotImplementedError
            else:
                replica_data=torch.cat([raw_data for _ in range(args.timesteps)],0)
                data = SpikeTensor(replica_data, args.timesteps,scale_factor=1)


            # theory error
            module_names={module:name for name,module in model.named_modules()}
            layer_outputs={}
            def layer_hook(module,input,output):
                layer_outputs[module_names[module]]=output

            for name,module in model.named_modules():
                if isinstance(module, SpikeConv2d):
                    module.register_forward_hook(layer_hook)
            output = model(data)

            spike_outputs=copy.deepcopy(layer_outputs)
            layer_outputs.clear()

            raw_output = model(raw_data)
            F1s=[]
            F2s=[]
            for name in layer_outputs:
                raw=layer_outputs[name]
                spike=spike_outputs[name]
                error=raw-spike.firing_ratio()
                F1=torch.mean(torch.abs(raw-spike.firing_ratio()))
                # F1=torch.norm(error,1)/np.prod(error.size())
                F2=torch.mean((raw-spike.firing_ratio())**2)
                # F2=torch.norm(error,2)/np.prod(error.size())
                print(f"F1 {F1} F2 {F2}")
                F1s.append(F1)
                F2s.append(F2)
            for name,module in model.named_modules():
                if isinstance(module, SpikeConv2d) or isinstance(module,SpikeAvgPool2d):
                    module._forward_hooks.clear()
            return F1s,F2s
            # exit()


            if spike_mode:
                output=output.to_float()
            elif hasattr(prediction_layer,'out_scales'):
                output=output*prediction_layer.out_scales.view(1,-1)


            target = target.to(device)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}'.format(
        top1=top1, top5=top5, batch_time=batch_time, losses=losses))
    # log to TensorBoard
    if train_writer is not None:
        train_writer.add_scalar('val_loss', losses.avg, epoch)
        train_writer.add_scalar('val_acc', top1.avg, epoch)

    return top1.avg, losses.avg


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('net_name',type=str)
    parser.add_argument('--resume', '-r', default=None, help='resume from checkpoint')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--timesteps', default=64, type=int)
    parser.add_argument('--input_poisson', action='store_true')
    parser.add_argument('--reset_mode', default='subtraction',type=str,choices=['zero','subtraction'])
    parser.add_argument('--half', default=False, type=bool)
    parser.add_argument('--weight_bits', default=4, type=int)
    args = parser.parse_args()
    args.dataset = 'CIFAR10'
    test_loader, val_loader, train_loader, train_val_loader=get_dataset(args)
    from build_network import get_net_by_name
    net=get_net_by_name(args.net_name)
    net.cuda()
    if args.resume:
        net.load_state_dict(torch.load(args.resume),False)
    # fuse the conv and bn
    # net.fuse_conv_bn()
    from ann2snn import trans_ann2snn
    device=torch.device('cuda')
    criterion=nn.CrossEntropyLoss()

    raw_acc, _ =validate(test_loader, net, device, criterion, 0, spike_mode=False)
    print("Testing the Non-Spike but weight transferred net")
    ann,snn=trans_ann2snn(net,val_loader,device,args.timesteps,args.weight_bits)

    # validate to get the stat for scale factor
    for m in snn.modules():
        if isinstance(m,SpikeReLU):
            m.quantize=True
    validate(test_loader,snn,device,criterion,0,spike_mode=False)
    # net.set_spike_mode()

    Xs = np.arange(1, 7, 1)
    accs = []
    for timesteps in 2 ** Xs:
        args.timesteps = timesteps
        ann,snn=trans_ann2snn(ann, val_loader, device, args.timesteps, args.weight_bits)
        acc, loss = validate(test_loader, snn, device, criterion, 0, spike_mode=True)
        accs.append(acc)
    plt.figure(figsize=(4, 3))
    plt.plot(Xs[:len(accs)], accs, label='SNN')
    plt.hlines(raw_acc, Xs.min(), Xs.max(), linestyles='--', label='ANN')
    plt.xlabel('timesteps')
    plt.ylabel('accuracy')
    plt.xticks(Xs, 2 ** Xs)
    plt.legend(loc='lower right')
    # plt.title(args.net_name)
    plt.show()
