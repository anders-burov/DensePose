import torch
import torch.nn as nn

class Siamese_Net(nn.Module):
    # Siamese-izes a given network.
    def __init__(self, net):
        # arguments:
        # net: should be constructed already
        super(Siamese_Net, self).__init__()
        # TODO assert is instance of nn module.
        self.net = net
    
    def forward_once(self, input):
        return self.net.forward(input)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2