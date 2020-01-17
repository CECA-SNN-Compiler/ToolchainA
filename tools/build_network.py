from models.testnet import *


def get_net_by_name(net_name):
    if 'testnet' in net_name:
        net=eval('TestNet%s()'%net_name[-1])
    elif 'testfull' in net_name:
        channel=[56,37,28][int(net_name[-1])-2]
        net=eval('TestFull(%s)'%channel)
    elif 'testres' in net_name:
        net=eval('TestResNet%s()'%net_name[-1])
    elif net_name=='vgg16':
        from models.vggv2 import vgg16
        net=vgg16()
    else:
        raise NotImplementedError
    return net
