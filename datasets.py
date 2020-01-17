import torch

import numpy as np
import os
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets


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
        if '100' in args.dataset:
            normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                             std=[0.2673, 0.2564, 0.2762])
        else:
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2470, 0.2435, 0.2616])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
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
        n_val=5000
        train_set.data=train_set.data[n_val:]
        train_set.targets=train_set.targets[n_val:]
        val_set.data=val_set.data[:n_val]
        val_set.targets=val_set.targets[:n_val]

        train_val_loader=torch.utils.data.DataLoader(
            train_val_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader=torch.utils.data.DataLoader(
            val_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            set(root, train=False, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise NotImplementedError
    return test_loader,val_loader,train_loader,train_val_loader

