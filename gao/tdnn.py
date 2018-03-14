#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Description:
The model here implements a generalized version of the TDNN based on the model descriptions given in [1],[2].
In the description given by Waibel et al. a TDNN uses the full context specified. For example: if the delay specified is N = 2, the model uses the current frame and frames at delays of 1 and 2.
In the description given by Peddinti et al. the TDNN only looks at the farthest frames from the current frame as specified by the context parameter. The description in section 3.1 of the paper discusses the differences between their implementation and Waibel et al.
The TDNN implemented here allows for the usage of an arbitrary context which shall be demonstrated in the usage code snippet.

Usage:
For the model specified in the Waibel et al. paper, the first layer is as follows:
context = [0,2]
input_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=True)

# For the model specified in the Peddinti et al. paper, the second layer is as follows (taking arbitrary I/O dimensions since it's not specified):
context = [-1,2]
input_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=False)

# You may also use any arbitrary context like this:
context = [-11,0,5,7,10]
nput_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=False)
# The above will convole the kernel with the current frame, 11 frames in the past, 5, 7, and 10 frames in the future.
output = net(input) # this will run a forward pass
'''

from torch.nn import Parameter
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import math

"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""

class TDNN(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context = True):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TDNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context,full_context)
        self.register_buffer('context',t.LongTensor(context))
        self.full_context = full_context
        stdv = 1./math.sqrt(input_dim)
        self.kernel = nn.Parameter(t.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
        self.bias = nn.Parameter(t.Tensor(output_dim).normal_(0,stdv))
        # self.cuda_flag = False

    def forward(self,x):
        """
        x is one batch of data
        x.size(): [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features
        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        # if type(self.bias.data) == torch.cuda.FloatTensor and self.cuda_flag == False:
        #     self.context = self.context.cuda()
        #     self.cuda_flag = True
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        return F.relu(conv_out)

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context,
        one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        input_size = x.size()
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_sequence_length, input_dim] = input_size
        x = x.transpose(1,2).contiguous()

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        xs = Variable(self.bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))

        # Perform the convolution with relevant input frames
        for c, i in enumerate(valid_steps):
            features = t.index_select(x, 2, context+i)
            xs[:,:,c] = F.conv1d(features, kernel, bias = bias)[:,:,0]
        return xs

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1*context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)



class TDNN1(nn.Module):
    def __init__(self, kernels, input_embed_size, bias=False):
        """
        :param kernels: array of pairs (width, out_dim)
        :param input_embed_size: size of input. conv kernel has shape of [out_dim_{i}, input_embed_size, width_{i}]
        :param bias: whether to use bias when convolution is performed
        """
        super(TDNN1, self).__init__()

        self.input_embed_size = input_embed_size

        self.kernels = nn.ParameterList([Parameter(t.Tensor(out_dim, input_embed_size, kW).normal_(0, 0.05))
                                         for kW, out_dim in kernels])

        self.use_bias = bias

        if self.use_bias:
            self.biases = nn.ParameterList([Parameter(t.Tensor(out_dim).normal_(0, 0.05))
                                            for _, out_dim in kernels])

    def forward(self, x):
        """
        :param x: tensor with shape [batch_size, max_seq_len, max_word_len, char_embed_size]
        :return: tensor with shape [batch_size, max_seq_len, depth_sum]
        applies multikenrel 1d-conv layer along every word in input with max-over-time pooling
            to emit fixed-size output
        """

        input_size = x.size()
        input_size_len = len(input_size)

        assert input_size_len == 4, \
            'Wrong input rang, must be equal to 4, but {} found'.format(input_size_len)

        [batch_size, seq_len, max_word_len, _] = input_size

        # leaps with shape
        x = x.view(-1, max_word_len, self.input_embed_size).transpose(1, 2).contiguous()

        xs = [F.relu(F.conv1d(x, kernel, bias=self.biases[i] if self.use_bias else None))
              for i, kernel in enumerate(self.kernels)]
        xs = [x.max(2)[0].squeeze(2) for x in xs]

        x = t.cat(xs, 1)
        x = x.view(batch_size, seq_len, -1)

        return x