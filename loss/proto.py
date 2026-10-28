#! /usr/bin/python
# -*- encoding: utf-8 -*-
## Re-implementation of prototypical networks (https://arxiv.org/abs/1703.05175).
## Numerically checked against https://github.com/cyvius96/prototypical-network-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from loss.ap_loss import lossFunction


class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = False

        self.criterion  = torch.nn.CrossEntropyLoss()
        self.speaker_loss = lossFunction(init_w=10.0, init_b=-5.0)

        print('Initialised Prototypical Loss')

    def forward(self, x):

        assert x.size()[1] >= 2
        
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output  = -1 * (F.pairwise_distance(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))**2)
        label   = torch.from_numpy(numpy.asarray(range(0,stepsize))).cuda()
        nloss   = self.criterion(output, label)
        loss1 = self.speaker_loss(x)

        #return nloss
        return 0.1*nloss + loss1