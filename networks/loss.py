import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    simple contrastive loss
    if label == 0: means the input x1 and x2 must similar
    In contrast, label ==1: mean the input x1 and x2 are not similar
    """
    def __init__(self, margin=5.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, x1, x2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(x1, x2)
        loss = torch.mean((1-label)*torch.pow(euclidean_distance, 2) +
                          (label)*torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

class TripletLoss_v2(nn.Module):
    """
    The TripletLoss_v1 directly compute the MSE distance between anchor, negative, and positive feature in feature space.
    and the each feature size is (2048,), when computing distance directly. It seems that every channel(dim) as equal, which
    may negatively effect.
    Thus, the TripleLoss_v2 is tried to go through linear layers to automaticly select and reduce dim of features.
    """
    def __init__(self, margin=1.0, feature_size=2048):
        super(TripletLoss_v2, self).__init__()
        self.linear = nn.Linear(feature_size, feature_size//8)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True),
        self.triplet_v1 = nn.TripletMarginLoss(margin=margin, p=2, eps=1e-7)
        
        
    def forward(self, feature_A, feature_P, feature_N):  
        # print(feature_A.size())
        # print(feature_P.size())
        feature_a = self.linear(feature_A)
        feature_p = self.linear(feature_P)
        feature_n = self.linear(feature_N)
        loss = self.triplet_v1(feature_a.squeeze(1), feature_p.squeeze(1), feature_n.squeeze(1))
        return loss
    
    
             