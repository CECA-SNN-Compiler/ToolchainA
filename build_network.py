from models.testnet import *


def get_net_by_name(net_name):
    if 'testnet' in net_name:
        net=eval('TestNet%s()'%net_name[-1])
    elif net_name=='vgg16':
        from models.vggv1 import VGG16V1
        net=VGG16V1()
    else:
        raise NotImplementedError
    return net