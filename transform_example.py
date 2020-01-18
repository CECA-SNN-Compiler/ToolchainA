import torch
import torch.nn as nn
from spike_layers import SpikeReLU,SpikeConv2d,SpikeLinear,SpikeAvgPool2d,spike_avg_pooling
from spike_tensor import SpikeTensor
from quantization import quantize_finetune
import datasets
from  ann2snn import trans_ann2snn
from validation import validate_snn,validate_ann
import argparse

class ExampleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # SpikeConv2d can be Conv2d in
        self.conv1 = SpikeConv2d(3, 8, 5, bias=False)
        self.relu1=SpikeReLU()
        self.conv2 = SpikeConv2d(8, 16, 3, bias=False)
        self.relu2 = SpikeReLU()
        self.conv3 = SpikeConv2d(16, 32, 3, bias=False)
        self.relu3 = SpikeReLU()
        self.fc1 = SpikeLinear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = spike_avg_pooling(out, 2)
        out = self.relu2(self.conv2(out))
        out = spike_avg_pooling(out, 2)
        out = self.relu3(self.conv3(out))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

if __name__=='__main__':
    # parse args
    parser=argparse.ArgumentParser()
    parser.add_argument('--load',type=str,help='the location of the trained weights')
    parser.add_argument('--dataset',default='CIFAR10',type=str,help='the location of the trained weights')
    parser.add_argument('--save_file',default="./out_snn.pth",type=str,help='the output location of the transferred weights')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--timesteps', default=64, type=int)
    parser.add_argument('--reset_mode', default='subtraction', type=str, choices=['zero', 'subtraction'])
    parser.add_argument('--weight_bitwidth', default=4, type=int,help='weight quantization bitwidth')
    parser.add_argument('--finetune_lr', default=0.005, type=float,help='finetune learning rate')
    parser.add_argument('--finetune_epochs', default=60, type=int,help='finetune epochs')
    parser.add_argument('--finetune_wd', default=5e-4, type=float,help='finetune weight decay')
    parser.add_argument('--finetune_momentum', default=0.9, type=float,help='finetune momentum')

    args=parser.parse_args()

    # Build Model
    net=ExampleNet()
    if args.load:
        net.load_state_dict(torch.load(args.load))
    net.cuda()

    # Preparing for train
    test_loader, val_loader, train_loader, train_val_loader=datasets.get_dataset(args)
    criterion=nn.CrossEntropyLoss()
    device=torch.device("cuda")

    # Weight quantization
    qnet=quantize_finetune(net,train_val_loader,criterion,device,args)
    qnet_top1,qnet_loss=validate_ann(qnet,test_loader,device,criterion)

    # SNN transform
    snn=trans_ann2snn(qnet,train_val_loader,device,args.timesteps,args.weight_bitwidth)

    # Test the results
    snn_top1,snn_loss=validate_snn(snn,test_loader,device,criterion,args.timesteps)

    # Save the SNN
    torch.save(snn,args.save_file)
    print("Save the SNN in {}".format(args.save_file))
