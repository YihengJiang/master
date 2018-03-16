import torch
from torch.autograd import Variable
import torch.nn as nn

class weight_pool(torch.nn.Module):
    def __init__(self):
        super(weight_pool, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, x, u):
        b, c, h, w = x.data.size()
        at = self.softmax(u.resize(b, w))
        x = x.resize(b, c, w)
        at = at.resize(b, w, h)
        out = x.bmm(at)
        return out.resize(b, c, 1, 1)


def test():
    x = Variable(torch.rand((2,5,1,3)).cuda())
    u = Variable(torch.rand((2,1,1,3)).cuda())
    layer = weight_pool()
    out = layer.forward(x,u)
    return out

if __name__ == '__main__':
    print(test())