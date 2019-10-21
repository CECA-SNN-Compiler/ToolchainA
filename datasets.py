import torch


from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import numpy as np
from functools import partial
from torch.utils.data import DataLoader
import os
import pickle
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#####################
## Augmentations
#####################

class GPUDataSet():
    def __init__(self,dataset,train=True,batch_size=128):
        print("build GPUDataset",dataset,train,batch_size)
        if dataset=='CIFAR10':
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            raw_set=datasets.CIFAR10('/datasets/torch', train=train, transform=transform)
            loader=DataLoader(raw_set,batch_size=512,num_workers=8)
        else:
            raise NotImplementedError
        self.train=train
        self.batch_size=batch_size
        datas=[]
        labels=[]
        for data,label in loader:
            datas.append(data)
            labels.append(label)
        self.datas=torch.cat(datas).cuda()
        if train:
            self.datas=F.pad(self.datas,pad=[8,8,8,8])
        self.labels=torch.cat(labels).cuda().view(-1)

    def __iter__(self):
        inds=np.arange(0,len(self.datas),self.batch_size)

        if self.train:
            perm_inds = torch.randperm(len(self.datas))
            self.datas = self.datas[perm_inds]
            self.labels = self.labels[perm_inds]

            # augmentation
            stride=2
            N=(8*2)//stride
            M=len(self.datas)//(N*N)
            new_datas=[]
            x=0
            for i in range(N):
                for j in range(N):
                    cropped=self.datas[x*M:(x+1)*M,:,i*stride:i*stride+32,j*stride:j*stride+32]
                    new_datas.append(cropped)
                    x+=1
            new_datas=torch.cat(new_datas)
            torch.flip(new_datas[len(self.datas) // 2:], [2])
            np.random.shuffle(inds)
        else:
            new_datas = self.datas
        new_labels = self.labels
        for ind in inds:
            if ind+self.batch_size>len(new_datas):
                continue
            yield new_datas[ind:ind+self.batch_size].detach(),new_labels[ind:ind+self.batch_size].detach()
    def __getitem__(self, item):
        return self.datas[item],self.labels[item]

    def __len__(self):
        return len(self.datas)//self.batch_size

def half(x):
    return x.half()

def get_dataset(args):
    # Preparing Datasets

    if args.dataset == "MNIST":
        raise NotImplementedError
        kwargs = {'num_workers': 8, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/datasets/torch', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/datasets/torch', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)


    elif "CIFAR" in args.dataset:
        # Data loading code
        if '10' in args.dataset:
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2470, 0.2435, 0.2616])
        else:
            normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                             std=[0.2673, 0.2564, 0.2762])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]+([half] if args.half else []))

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]+([half] if args.half else []))
        kwargs = {'num_workers': 4, 'pin_memory': False}
        root='/datasets/torch'
        if not os.path.exists(root):
            root='./data'
        if args.dataset=='CIFAR10':
            set=datasets.CIFAR10
        else:
            set=datasets.CIFAR100

        train_val_set=set(root, train=True, download=True, transform=transform_train)
        rand_inds = np.arange(len(train_val_set))
        np.random.seed(3)
        np.random.shuffle(rand_inds)
        val_set=copy.deepcopy(train_val_set)
        train_set=copy.deepcopy(train_val_set)
        train_set.data=train_set.data[5000:]
        train_set.targets=train_set.targets[5000:]
        val_set.data=train_set.data[:5000]
        val_set.targets=train_set.targets[:5000]

        train_val_loader=torch.utils.data.DataLoader(
            train_val_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader=torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            set(root, train=False, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset=='ImageNet':
        root='/media/nvme0/imagenet'
        version=1
        if not os.path.exists(root):
            root = './data/imagenet'
        cache_file='/tmp/S2DI_{}.pkl'.format('h' if args.half else 'f')
        if os.path.exists(cache_file):
            try:
                _version,train_val_set,test_set=pickle.load(open(cache_file,'rb'))
                if _version != version:
                    os.remove(cache_file)
            except:
                os.remove(cache_file)

        if not os.path.exists(cache_file):
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform_train = transforms.Compose([
                transforms.Scale(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]+([half] if args.half else []))

            transform_test = transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),

            ]+([half] if args.half else []))
            train_val_set = ImageFolder(root + '/train/', transform_train)
            test_set = ImageFolder(root + '/val/', transform_test)
            pickle.dump([version,train_val_set,test_set],open(cache_file, 'wb'))
        train_set=copy.deepcopy(train_val_set)
        del train_set.imgs[-int(len(train_set)/10):]
        val_set=copy.deepcopy(train_val_set)
        del val_set.imgs[:-int(len(train_set)/10)]
        train_val_loader = DataLoader(train_val_set, batch_size=args.batch_size, shuffle=True, num_workers=torch.get_num_threads())
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=torch.get_num_threads())
        val_loader=DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=torch.get_num_threads())
        # val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=torch.get_num_threads())
    else:
        raise NotImplementedError
    return test_loader,val_loader,train_loader,train_val_loader

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--test_batch_size', default=500, type=int)
    parser.add_argument('--half', default=0, type=int)
    args = parser.parse_args()
    args.dataset = 'CIFAR100'

    import matplotlib.pyplot as plt
    test,val,train=get_dataset(args)
    for data,input in test:
        Y,X=np.histogram(data.numpy())
        print(Y,X)
        plt.plot(X[:-1],Y)
        break
    for data,input in train:
        Y, X=np.histogram(data.numpy())
        print(Y,X)
        plt.plot(X[:-1], Y)
        break
    plt.show()
