"""
2d Convolution Module.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * \
            np.random.randn(self.out_channels, self.in_channels,
                            self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################

        def compute_output_dimensions(x, padding, window, stride):
            return int(1 + (H + 2 * padding - window) / stride)
        N, C, H, W = x.shape
        pad = self.padding

        F, _, HH, WW = self.weight.shape
        output_height = compute_output_dimensions(
            x, self.padding, HH, self.stride)
        output_width = compute_output_dimensions(
            x, self.padding, WW, self.stride)

        out = np.zeros((N, F, output_height, output_width))
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

        def compute_convolution(curr,i,out):
            for weight_index,weight in enumerate(self.weight):
                '''Iterate from 0 to the padded shape - filter height, in intervals of our stride variable '''
                for curr_x_position in range(0, x_pad.shape[2]+1-self.kernel_size, self.stride):
                    out_y = curr_x_position - self.stride
                    '''Iterate from 0 to the padded shape - filter width, in intervals of our stride variable '''
                    for curr_y_position in range(0, x_pad.shape[2]+1-self.kernel_size, self.stride):
                        '''take a slice of current data, everything along first dimension, and only x<x+filter size &&
                            y < y+ filter size '''
                        window_slice = curr[:, curr_x_position:(curr_x_position + self.kernel_size), curr_y_position:(curr_y_position + self.kernel_size)]
                        '''multiply and sum weights in our slice, computing the convolution scalar
                            for that particular index. update the output variable'''
                        out[i, weight_index, int(curr_x_position/self.stride),
                            int(curr_y_position/self.stride)] = np.sum(np.multiply(weight, window_slice)) + self.bias[weight_index]

        for i, value in enumerate(x_pad):
            compute_convolution(x_pad[i],i,out)
            #np.apply_along_axis(compute_convolution, 0, b)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        x, w, b = self.cache, self.weight, self.bias
        N, C, H, W = x.shape
        F, _, HH, WW = self.weight.shape
        stride = self.stride
        pad = self.padding

        # init
        dx = np.zeros((N, C, H, W))
        dw = np.zeros((F, C, HH, WW))
        db = dout.sum(0).sum(1).sum(1)
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                       mode='constant', constant_values=(0))  # add padding to x
        dx_pad = np.zeros(x_pad.shape)

        # backward pass
        for i in range(N):
            for j in range(F):
                for q in range(0, 1 + H + 2 * pad - HH, stride):
                    for p in range(0, 1 + W + 2 * pad - WW, stride):
                        w_tmp = w[j, :, :, :]
                        x_tmp = x_pad[i, :, q:(q + HH), p:(p + WW)]
                        dw[j, :, :, :] += x_tmp * \
                            dout[i, j, int(q/stride), int(p/stride)]
                        dx_pad[i, :, q:(q + HH), p:(p + WW)] += w_tmp * \
                            dout[i, j, int(q/stride), int(p/stride)]

        dx = dx_pad[:, :, pad:(pad + H), pad:(pad + W)]
        self.dx = dx
        self.dw = dw
        self.db = db

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################