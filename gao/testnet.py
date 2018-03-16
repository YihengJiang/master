#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class net(nn.Module):
    def __init__(self, num_classes):
        super(net, self).__init__()
        self.conv1=nn.Conv1d(39,64,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm1d(64)
        self.pool1=nn.MaxPool1d(kernel_size=2,stride=2)

        self.conv2=nn.Conv1d(64,128,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm1d(128)
        self.pool2=nn.MaxPool1d(kernel_size=2,stride=2)

        self.conv3=nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm1d(256)
        self.pool3=nn.MaxPool1d(kernel_size=2,stride=2)

        self.conv4=nn.Conv1d(256,512,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm1d(512)
        self.pool4=nn.MaxPool1d(kernel_size=2,stride=2)



        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, num_classes, bias=True)
        self.gap = nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x=torch.squeeze(x,1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.pool4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
