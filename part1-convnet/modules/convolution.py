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

        def compute_output_dimensions(input_dim, padding, window, stride):
            return 1 + (input_dim + 2 * padding - window) // stride
        N = x.shape[0]
        output_height = compute_output_dimensions(
            x.shape[2], self.padding, self.kernel_size, self.stride)
        output_width = compute_output_dimensions(
            x.shape[3], self.padding, self.kernel_size, self.stride)

        out = np.zeros((N, self.out_channels, output_height, output_width))
        padded_x = np.pad(
            x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))

        def compute_convolution(curr, i, out):
            for weight_index, weight in enumerate(self.weight):
                '''Iterate from 0 to the padded shape - filter height, in intervals of our stride variable '''
                out_x = 0
                for curr_x_position in range(0, padded_x.shape[2]+1-self.kernel_size, self.stride):
                    '''Iterate from 0 to the padded shape - filter width, in intervals of our stride variable '''
                    out_y = 0
                    for curr_y_position in range(0, padded_x.shape[2]+1-self.kernel_size, self.stride):
                        '''take a slice of current data, everything along first dimension, and only x<x+filter size &&
                            y < y+ filter size '''
                        window_slice = curr[:, curr_x_position:(
                            curr_x_position + self.kernel_size), curr_y_position:(curr_y_position + self.kernel_size)]
                        '''multiply and sum weights in our slice, computing the convolution scalar
                            for that particular index. update the output variable'''
                        out[i, weight_index, out_x,
                            out_y] = np.sum(np.multiply(weight, window_slice)) + self.bias[weight_index]
                        out_y = out_y+1
                    out_x = out_x + 1

        for i, value in enumerate(padded_x):
            compute_convolution(padded_x[i], i, out)
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
        x = self.cache
        out_dim_x = self.padding+x.shape[2]
        out_dim_y = self.padding+x.shape[3]
        dw = np.zeros((self.weight.shape[0], x.shape[1], self.kernel_size, self.kernel_size))
        pad_x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))  # add padding to x
        dx = np.zeros(pad_x.shape)


        def compute_backward_pass(i,curr):
            for weight_index, weight in enumerate(self.weight):
                '''Iterate across all weights '''
                out_x = 0
                for curr_x_pos in range(0, curr.shape[2] + 1 - self.kernel_size, self.stride):
                    '''Iterate from 0 to the padded shape - filter height, in intervals of our stride variable '''
                    out_y = 0
                    for curr_y_pos in range(0, curr.shape[2] + 1 - self.kernel_size, self.stride):
                        '''update padded dx vector with respect to i (the index of the current image we are looking at)'''
                        dx[i, :, curr_x_pos:(curr_x_pos + self.kernel_size), curr_y_pos:(curr_y_pos + self.kernel_size)] += weight * \
                            dout[i, weight_index, out_x, out_y]
                        '''take a slice of current data, everything along first dimension, and only x<x+filter size &&
                            y < y+ filter size '''
                        window_slice = curr[:, curr_x_pos:(curr_x_pos + self.kernel_size), curr_y_pos:(curr_y_pos + self.kernel_size)]
                        '''update current weight vector by adding slice to dout of curr slice'''
                        dw[weight_index] += window_slice * \
                            dout[i, weight_index, out_x, out_y]
                        out_y += 1
                    out_x += 1

        for i,value in enumerate(pad_x):
            '''Compute backward pass for each image in batch'''
            compute_backward_pass(i,value)
        self.dw = dw
        '''resize dx so it removes extra padding dimensions. if 0, it does nothing'''
        self.dx = dx[:, :, self.padding:out_dim_x, self.padding:out_dim_y]
        '''sum across all inputs, then width + height. final output is separated by channel'''
        self.db = dout.sum(0).sum((1,2))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
