from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import sys
import pickle
import collections
from numpy import linalg as LA
import copy

import datetime
import operator
import random
from .prune_base import PruneBase
from .admm import ADMM
from .retrain import Retrain
from . import  utils_pr

# from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc

admm = None


class MultiLevelADMM(ADMM):
    def __init__(self, args, model, logger=None):
        self.frozen_weights = {}
        super(MultiLevelADMM, self).__init__(args, model, logger, False)

        assert(args.sp_load_frozen_weights is not None)
        frozen_ratios = self.configs["frozen_ratios"] if "frozen_ratios" in self.configs else None
        self.frozen_weights = utils_pr.get_frozen_weights(model,
            args.sp_load_frozen_weights, self.prune_ratios, frozen_ratios)

        # do init again to fix the issue that the frozen_weights
        # were not available when first doing the pruning
        if self.args.sp_load_prune_params is None:
            super(MultiLevelADMM, self).init()


    def apply_masks(self):
        utils_pr.apply_masks(self.model, self.frozen_weights)

    def prune_weight(self, name, weight, prune_ratio, first):
        W = weight.clone().detach().cuda()
        # import pdb; pdb.set_trace()
        if name in self.frozen_weights:
            # calculate new prune ratio
            total_num = W.numel()
            extra_num = torch.sum(self.frozen_weights[name]['mask'].float()).\
                detach().cpu().numpy().item()
            frozen_nnz = total_num - extra_num
            target_nnz = total_num * (1 - prune_ratio)
            remaining_nnz = target_nnz - frozen_nnz
            prune_ratio = (total_num - remaining_nnz) * 1.0 / total_num
            # if it is too close to 1, just prune everything
            prune_ratio = prune_ratio if prune_ratio < 0.999 else 1.0
            prune_ratio = prune_ratio if prune_ratio > 0.001 else 0.0
            W = W * self.frozen_weights[name]['mask'].type(W.dtype)
        else:
            self.logger.warning("Weight {} is not frozened, please check"
                .format(name))

        new_weight = super(MultiLevelADMM, self).prune_weight(
                name, W, prune_ratio, first)

        if name in self.frozen_weights:
            new_weight = utils_pr.update_one_frozen_weight(
                    new_weight,
                    self.frozen_weights[name]['mask'],
                    self.frozen_weights[name]['weight'])
        return new_weight
