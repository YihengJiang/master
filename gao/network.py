#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class net(nn.Module):
    def __init__(self, kernel_sizes=[5, 3, 3, 3, 1, 1], channels=[64, 128, 256, 512, 512, 128, 128],
                 num_classes=16,in_channel=1):
        super(net, self).__init__()
        # self.filter_sizes = filter_sizes
        # self.channels = channels
        # self.dilation = dilation
        # self.num_classes = num_classes
        # self.ndim = ndim
        self.layer_num=5
        self.convs=[]
        self.bns=[]
        self.poolings=[]
        channels.insert(0,in_channel)
        kernel_size=[[2,3],[2,3],[2,2],[2,2],[2,2]]
        for i in range(self.layer_num):
            conv=nn.Conv2d(channels[i],channels[i+1],kernel_sizes[i],1,int(np.floor(kernel_sizes[i]/2)))
            setattr(self,"conv%i"%i,conv)
            self.convs.append(conv)
            bn=nn.BatchNorm2d(channels[i+1])
            setattr(self,"bn%i"%i,bn)
            self.bns.append(bn)
            pooling=nn.MaxPool2d(kernel_size[i])
            setattr(self,"pooling%i"%i,pooling)
            self.poolings.append(pooling)
        # self.pooling1 = nn.MaxPool2d(kernel_size=[2,3])
        # self.pooling2 = nn.MaxPool2d(kernel_size=[2,3])
        # self.pooling3 = nn.MaxPool2d(kernel_size=[2,3])
        # self.pooling4 = nn.MaxPool2d(kernel_size=[2,2])
        # self.pooling5 = nn.MaxPool2d(kernel_size=[2,2])
        # self.conv1 = nn.Conv2d(in_channel, channels[0], kernel_size=(filter_sizes[0]), padding=2, bias=True,
        #                        dilation=dilation[0])
        # self.bn1 = nn.BatchNorm2d(channels[0])
        # # self.dropout1 = nn.Dropout2d(p=0.5)
        # self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=(filter_sizes[1]), padding=1,
        #                        bias=True, dilation=dilation[1])
        # self.bn2 = nn.BatchNorm2d(channels[1])
        # # self.dropout2 = nn.Dropout2d(p=0.5)
        # self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=(1, filter_sizes[2]),padding=1,
        #                        bias=True, dilation=dilation[2])
        # self.bn3 = nn.BatchNorm2d(channels[2])
        # # self.dropout3 = nn.Dropout2d(p=0.5)
        # self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=(1, filter_sizes[3]),  padding=1,
        #                        bias=True, dilation=dilation[3])
        # self.bn4 = nn.BatchNorm2d(channels[3])
        # # self.dropout4 = nn.Dropout2d(p=0.5)
        # self.conv5 = nn.Conv2d(channels[3], channels[4], kernel_size=(1, filter_sizes[4]), padding=1,
        #                        bias=True, dilation=dilation[4])
        # self.bn5 = nn.BatchNorm2d(channels[4])
        # self.dropout5 = nn.Dropout2d(p=0.5)
        # statics pooling
        # self.statics_pooling = statics_pooling.statics_pooling(dim=3)
        # self.statics_pooling = nn.AdaptiveAvgPool2d(1)
        # self.bilinear = bilinear()
        # embeding
        self.embeding1 = nn.Linear(channels[4]*1*3, channels[5])
        self.bn6 = nn.BatchNorm1d(channels[5])
        # self.dropout6 = nn.Dropout(p=0.5)
        self.embeding2 = nn.Linear(channels[5], channels[6])
        self.bn7 = nn.BatchNorm1d(channels[6])

        self.relu = nn.ReLU()
        self.fc = nn.Linear(channels[6], num_classes)
        # self.gap = nn.AdaptiveAvgPool2d(1)

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
            x=self.convs[i](x)
            x=self.bns[i](x)
            x=self.relu(x)
            x=self.poolings[i](x)

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out=self.pooling1(out)
        # # out = self.dropout1(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        # out = self.pooling2(out)
        # # out = self.dropout2(out)
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        # out = self.pooling3(out)
        # # out = self.dropout3(out)
        # out4 = self.conv4(out)
        # out = self.bn4(out4)
        # out = self.relu(out)
        # out = self.pooling4(out)
        # # out = self.dropout4(out)
        # out5 = self.conv5(out)
        # out = self.bn5(out5)
        # out = self.relu(out)
        # out = self.pooling5(out)
        # # out = self.dropout5(out)

        # out = self.statics_pooling(out)
        # out = self.bilinear(out, out4)
        # out = self.bilinear.ssqrt(out)
        # out = self.bilinear.l2norm(out)
        x = x.view(x.size(0), -1)

        embeding1 = self.embeding1(x)
        x = self.bn6(embeding1)
        x = F.relu(x)
        # out = self.dropout6(out)

        embeding2 = self.embeding2(x)
        x = self.bn7(embeding2)
        x = F.relu(x)

        out = self.fc(x)
        # out = out.view(out.size(0), -1)

        return out, embeding1, embeding2

if __name__ == '__main__':

    n=net()
    print(n)
