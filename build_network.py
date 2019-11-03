

def get_net_by_name(net_name):
    if net_name=='testnet':
        from models.testnet import TestNet
        net = TestNet()
    elif net_name=='testnet2':
        from models.testnet2 import TestNet2
        net = TestNet2()
    elif net_name=='testnet3':
        from models.testnet3 import TestNet3
        net = TestNet3()
    else:
        raise NotImplementedError
    return net