#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
# from tuneThreshold import tuneThresholdfromScore
import random
from loss.ap_loss import lossFunction

class LossFunction(nn.Module):

    def __init__(self, hard_rank=-1, hard_prob=0, margin=0.3, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        
        self.hard_rank  = hard_rank
        self.hard_prob  = hard_prob
        self.margin     = margin
        self.speaker_loss = lossFunction(init_w=10.0, init_b=-5.0)

        print('Initialised Triplet Loss')

    def forward(self, x,):

        assert x.size()[1] == 2
        
        out_anchor      = F.normalize(x[:,0,:], p=2, dim=1)
        out_positive    = F.normalize(x[:,1,:], p=2, dim=1)
        stepsize        = out_anchor.size()[0]

        output      = -1 * (F.pairwise_distance(out_anchor.unsqueeze(-1),out_positive.unsqueeze(-1).transpose(0,2))**2)

        negidx      = self.mineHardNegative(output.detach())

        negative_list = []
        for idx, similarity in enumerate(negidx):
            negative_list.append(out_positive[idx,similarity])
        out_negative = torch.stack(negative_list,dim=0).unsqueeze(1)



        ## calculate distances
        pos_dist    = F.pairwise_distance(out_anchor,out_positive)
        neg_dist    = F.pairwise_distance(out_anchor,out_negative)

        ## loss function
        nloss   = torch.mean(F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin))
        loss1 = self.speaker_loss(x)

        #return nloss
        return 0.1 * nloss +  loss1

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Hard negative mining
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def mineHardNegative(self, output):

        negidx = []

        for idx, similarity in enumerate(output):

            simval, simidx = torch.sort(similarity,descending=True)

            if self.hard_rank < 0:

                ## Semi hard negative mining

                semihardidx = simidx[(similarity[idx] - self.margin < simval) &  (simval < similarity[idx])]

                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))

            else:

                ## Rank based negative mining
                
                simidx = simidx[simidx!=idx]

                if random.random() < self.hard_prob:
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    negidx.append(random.choice(simidx))

        return negidx