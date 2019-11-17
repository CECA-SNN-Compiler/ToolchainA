

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
    elif net_name=='testnet4':
        from models.testnet4 import TestNet4
        net = TestNet4()
    elif net_name=='vgg16':
        from models.vggv1 import VGG16V1
        net=VGG16V1()
    else:
        raise NotImplementedError
    return net