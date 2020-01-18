import os
from datasets import get_dataset
import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import GPUtil
from tools.build_network import get_net_by_name
from ann2snn import trans_ann2snn
import matplotlib.pyplot as plt
import seaborn as sns
from validation import validate_ann,validate_snn
from quantization import quantize_finetune


min_mem_gpu = np.argmin([_.memoryUsed for _ in GPUtil.getGPUs()])
print("selecting GPU {}".format(min_mem_gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = str(min_mem_gpu)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('net_name',type=str)
    parser.add_argument('--dataset',default='CIFAR10',type=str,help='the location of the trained weights')
    parser.add_argument('--base_lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--load', '-r', default=None, help='load from checkpoint')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--reset_mode', default='subtraction',type=str,choices=['zero','subtraction'])
    parser.add_argument('--weight_bitwidth', default=4, type=int, help='weight quantization bitwidth')
    parser.add_argument('--finetune_lr', default=0.005, type=float, help='finetune learning rate')
    parser.add_argument('--finetune_epochs', default=30, type=int, help='finetune epochs')
    parser.add_argument('--finetune_wd', default=5e-4, type=float, help='finetune weight decay')
    parser.add_argument('--finetune_momentum', default=0.9, type=float, help='finetune momentum')

    args = parser.parse_args()
    test_loader, val_loader, train_loader, train_val_loader=get_dataset(args)
    device=torch.device('cuda')

    # Build Model
    criterion=nn.CrossEntropyLoss()
    net=get_net_by_name(args.net_name)
    net.to(device)

    if args.load:
        net.load_state_dict(torch.load(args.load),False)

    # test the raw accuracy
    raw_acc, _ = validate_ann(net, test_loader, device, criterion)

    # Weight quantization
    qnet = quantize_finetune(net, train_val_loader, criterion, device, args)
    qnet_acc, qnet_loss = validate_ann(qnet, test_loader, device, criterion)

    accs=[]
    Xs=np.arange(1,8,1)
    for timesteps in 2**Xs:
        args.timesteps=timesteps
        snn = trans_ann2snn(qnet, train_val_loader, device, args.timesteps, args.weight_bitwidth)
        try:
            acc,loss=validate_snn(snn,test_loader,device,criterion,timesteps)
            accs.append(acc)
        except:
            pass
    sns.set_style('whitegrid')
    plt.figure(figsize=(5,4))
    plt.plot(Xs,accs,label='SNN')
    plt.hlines(raw_acc,Xs.min(),Xs.max(),linestyles='--',label='ANN')
    plt.hlines(qnet_acc,Xs.min(),Xs.max(),linestyles='--',colors='red',label='Quantized ANN')
    plt.xlabel('timesteps')
    plt.ylabel('accuracy')
    plt.xticks(Xs,2**Xs)
    plt.legend(loc='lower right')
    plt.title(args.net_name)
    plt.show()
