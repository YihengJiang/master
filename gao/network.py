#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class net(nn.Module):
    def __init__(self, conv_kernel_sizes, channels, num_classes, in_channel, pooling_kernel_sizes, gap):
        super(net, self).__init__()
        self.layer_num = len(conv_kernel_sizes)
        self.convs = []
        self.bns = []
        self.poolings = []
        channels.insert(0, in_channel)
        for i in range(self.layer_num):
            conv = nn.Conv2d(channels[i], channels[i + 1], conv_kernel_sizes[i], 1,
                             int(np.floor(conv_kernel_sizes[i] / 2)))
            setattr(self, "conv%i" % i, conv)
            self.convs.append(conv)

            bn = nn.BatchNorm2d(channels[i + 1])
            setattr(self, "bn%i" % i, bn)
            self.bns.append(bn)

            pooling = nn.MaxPool2d(pooling_kernel_sizes[i])
            setattr(self, "pooling%i" % i, pooling)
            self.poolings.append(pooling)

        # change the last pooling to gap in the convs
        gapPooling = nn.AdaptiveAvgPool2d(gap)
        setattr(self, "pooling%i" % 4, gapPooling)
        self.poolings.pop()
        self.poolings.append(gapPooling)
        # fc layer
        self.fc = nn.Linear(channels[5] * gap[0] * gap[1], num_classes)
        # relu
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.001)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = self.relu(x)
            x = self.poolings[i](x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        # out = out.view(out.size(0), -1)

        return out


if __name__ == '__main__':
    from gao.cnn_main import P
    n = net(P.net_conv_kernel_sizes, P.net_channels, P.net_num_classes, P.net_in_channel,
                        P.net_pooling_kernel_sizes, P.net_gap)
    print(n)
