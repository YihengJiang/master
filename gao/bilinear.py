from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class bilinear(torch.nn.Module):
    def __init__(self):
        super(bilinear, self).__init__()

    def forward(self, x, u):

        b, c, h, w = x.data.size()
        x = x.resize(b, c, h*w)

        b_u, c_u, h_u, w_u = u.data.size()
        u = u.resize(b_u, c_u, h_u*w_u)
        out = x.bmm(u.transpose(1,2))
        out = torch.div(out, h*w)
        return out.resize(b, c*c_u, 1, 1)
    def ssqrt(self, x):

        out = torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))
        return out
    def l2norm(self, x):
        eps = 1e-10
        norm = x.pow(2).sum(1).sqrt() + eps
        s = norm.size()
        norm = norm.resize(s[0], s[1], s[2], 1)
        x = x / norm.expand_as(x)
        return x

#
# def test():
#     x = Variable(torch.rand((2,5,4,2)).cuda())
#     u = Variable(torch.rand((2,3,4,2)).cuda())
#     layer = bilinear()
#     out = layer.forward(x,u)
#     return out
#
# if __name__ == '__main__':
#     print(test())