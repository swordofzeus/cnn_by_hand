"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        out = np.zeros((x.shape[0], x.shape[1], x.shape[2] //
                        self.stride, x.shape[3]//self.stride))

        def max_pool(i, val):
            for channel in range(x.shape[1]):
                '''Iterate over all channels in image'''
                out_x = 0
                '''Iterate for all x values between 0 and width - stride'''
                for curr_x_pos in range(0, x.shape[2] + 1 - self.kernel_size, self.stride):
                    out_y = 0
                    '''Iterate for all y values between 0 and height - stride'''
                    for curr_y_pos in range(0, x.shape[3] + 1 - self.kernel_size, self.stride):
                        '''Get current slice using kernal'''
                        window_slice = val[channel, curr_x_pos:(
                            curr_x_pos+self.kernel_size), curr_y_pos:(curr_y_pos+self.kernel_size)]
                        '''Take max over slice, and add value to output array'''
                        out[i, channel, out_x, out_y] = np.max(window_slice)
                        out_y += 1
                    out_x += 1

        for i, val in enumerate(x):
            max_pool(i, val)

        H_out = None
        W_out = None
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        self.dx = np.zeros(x.shape)

        def back_maxpool(i, val):
            for curr_channel in range(val.shape[0]):
                out_x = 0
                for curr_x_pos in range(0, x.shape[2] - self.kernel_size + 1, self.stride):
                    out_y = 0
                    for curr_y_pos in range(0, x.shape[3] - self.kernel_size + 1, self.stride):
                        window_slice = val[curr_channel, curr_x_pos:(
                            curr_x_pos+self.kernel_size), curr_y_pos:(curr_y_pos+self.kernel_size)]
                        h, w = np.unravel_index(
                            np.argmax(window_slice), (self.kernel_size, self.kernel_size))
                        self.dx[i, curr_channel, curr_x_pos+h, curr_y_pos +
                                w] = dout[i, curr_channel, out_x, out_y]
                        out_y += 1
                    out_x += 1
        for i, val in enumerate(x):
            back_maxpool(i, val)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
