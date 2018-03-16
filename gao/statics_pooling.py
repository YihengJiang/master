import torch
from torch.autograd import Variable

import torch.nn as nn

class statics_pooling(torch.nn.Module):
    def __init__(self, dim):
        super(statics_pooling, self).__init__()
        self.dim  = dim
    def forward(self, x):
        mean = x.mean(dim=self.dim)
        var =  x.var(dim=self.dim)
        return torch.cat((mean, var), dim=1)#efficient on channels


def ce():
    x = Variable(torch.rand((2,3,2,4)).cuda())
    layer = statics_pooling(2)
    out = layer.forward(x)
    return out

if __name__ == '__main__':
    o=ce()
    print(o)