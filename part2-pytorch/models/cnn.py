"""
Vanilla CNN model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        #32 output channels, akernel size of 7withstride 1andzeropadding
        self.conv_layer = nn.Conv2d(in_channels = 3 , out_channels = 32, kernel_size = 7,padding=0,stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear = nn.Linear(5408, 10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #print(x.shape)
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        #x = torch.flatten(x, start_dim=1)
        out = self.conv_layer(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = out.view(-1,out.shape[1]*out.shape[2]*out.shape[3])
        #print(x.shape)
        #exit()
        out = self.linear(out)
        outs = out
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs
