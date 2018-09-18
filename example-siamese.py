from __future__ import print_function
# import argparse
import torch
import torch.nn as nn
# # import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
from person_reid_nets import Siamese_Net
import resnet_custom
import torchvision.models as models
# from torchsummary import summary    # sigh not working on pyhon 2.7 Sep 2018

if __name__ == "__main__":
    net = resnet_custom.resnet18(input_channels=24*3)
    # print(summary(net, (24*3,224,224)))     # sigh not working on pyhon 2.7 Sep 2018
    net = Siamese_Net(net)
    T = torch.randn(2,24*3,224,224)
    T0 = torch.index_select(T, 0, torch.tensor([0]))
    print(T0.shape)
    o1, o2 = net.forward(T0,T0)
    print(type(o1), o1.shape)
    