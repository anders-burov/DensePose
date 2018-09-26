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
from torchsummary import summary 





if __name__ == "__main__":
    # net = resnet_custom.resnet18(input_channels=24*3, num_classes=256)
    #net = resnet_custom.ResNet(resnet_custom.BasicBlock, [1, 1, 1, 1], input_channels=24*3, num_classes=256)
    net = resnet_custom.ResNet(resnet_custom.BasicBlock, layers=[2, 2, 2, 2], 
                                inplanes=256, planes_of_layers=(256, 512, 1024, 2048), input_channels=24*3, num_classes=256)
    # net = resnet_custom.resnet152(input_channels=24*3, num_classes=256)
    # net = resnet_custom.ResNet(resnet_custom.BasicBlock, layers=[2, 2, 2, 2], 
    #                             inplanes=256, planes_of_layers=(128, 256, 512, 1024), input_channels=24*3, num_classes=256)
   
    print(net)
    net.eval()
    net.forward(torch.rand((1,24*3,224,224)))
    summary(net, input_size=(1,24*3,224,224), batch_size=1)


    # inplanes=64, planes_of_layers