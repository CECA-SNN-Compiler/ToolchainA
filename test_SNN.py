from datasets import get_dataset
from models.testnet import TestNet
import argparse
import torch
import numpy as np
import time
import torch.nn as nn
from spike_tensor import SpikeTensor

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

def validate(spike_mode, test_loader, model, device, criterion, epoch, train_writer=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test

            data = data.to(device)
            data=SpikeTensor(data,args.timesteps,False)
            if spike_mode:
                data.input_replica()
                data.is_spike=True

            output = model(data)

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
    parser.add_argument('--base_lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default=None, help='resume from checkpoint')
    # parser.add_argument('--resume', '-r', default="checkpoint/testnet_original_e10T2026_original.pth", help='resume from checkpoint')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--test_batch_size', default=256, type=int)
    parser.add_argument('--timesteps', default=100, type=int)
    parser.add_argument('--input_poisson', action='store_true')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--half', default=False, type=bool)
    args = parser.parse_args()
    args.dataset = 'CIFAR10'
    test_loader, val_loader, train_loader, train_val_loader=get_dataset(args)
    net = TestNet().cuda()
    if not args.input_poisson:
        net.conv1.process_input_im=True
    else:
        assert NotImplementedError
    if args.resume:
        net.load_state_dict(torch.load(args.resume),False)

    validate(False,test_loader,net,torch.device('cuda'),nn.CrossEntropyLoss(),0)

    net.spike_mode()
    validate(True,test_loader,net,torch.device('cuda'),nn.CrossEntropyLoss(),0)