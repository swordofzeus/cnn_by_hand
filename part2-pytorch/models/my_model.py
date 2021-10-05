"""
MyModel model.  (c) 2021 Georgia Tech

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


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        '''
            Gradually increase convolution window (*2)
            Gradually lower the linear layers to finally match output class (10)
        '''
        convolutional_kwargs = {'kernel_size': 2, 'padding':1}
        self.network = nn.Sequential(
            nn.Conv2d(3, 16,**convolutional_kwargs ),
            nn.Conv2d(16, 32, **convolutional_kwargs),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64,**convolutional_kwargs),
            nn.Conv2d(64, 128,**convolutional_kwargs),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256,**convolutional_kwargs),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(6400, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.network(x)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs
