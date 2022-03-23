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
from skimage.util.shape import view_as_windows

import time
import datetime
import operator
import random
from .prune_base import PruneBase
from .utils_pr import *


# from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc

# M:N pattern pruning
import heapq
from collections import defaultdict

admm = None


def prune_parse_arguments(parser):
    admm_args = parser.add_argument_group('Admm arguments')
    update_freq = admm_args.add_mutually_exclusive_group()
    update_freq.add_argument('--sp-admm-update-epoch', type=int,
                        help="how often we do admm update")
    update_freq.add_argument('--sp-admm-update-batch', type=int,
                        help="update admm after how many minibatches")
    admm_args.add_argument('--sp-admm-rho', type=float,
                        help="define rho for ADMM, overrides the rho specified in config file")
    admm_args.add_argument('--sp-admm-sparsity-type', type=str, default='gather_scatter',
                        help="define sp_admm_sparsity_type: [irregular, irregular_global, column,filter]")

    admm_args.add_argument('--sp-admm-lr', type=float, default=0.001,
                        help="define learning rate for ADMM, reset to sp_admm_lr every time U,Z is updated. Overrides the learning rate of the outside training loop")
    admm_args.add_argument('--admm-debug', dest='admm_debug', action='store_true',
                        help='debug mode of admm, print out some values (e.g., loss)')

    admm_args.add_argument('--sp-global-weight-sparsity', type=float, default=-1, help="Use global weight magnitude to prune, override the --sp-config-file")
    admm_args.add_argument('--sp-prune-threshold', type=float, default=-1.0, help="Used with --sp-global-weight-sparsity. Threshold of sparsity to prune. For example, if set this threshold = 0.1, then only prune layers with sparsity > 0.1 in a global fashion ")
    admm_args.add_argument('--sp-block-irregular-sparsity', type=str, default="(0,0)", help="blocked and irregular sparsity in block + irregular sparse pattern")


    # the following is for gather/scatter sparsity type
    admm_args.add_argument('--sp-admm-block', default="(1,)")
    admm_args.add_argument('--sp-admm-buckets-num', type=int, default=16)
    # this is not needed, should be calculated
    # admm_args.add_argument('--sp-admm-bucket-axis', type=int, default=1)
    admm_args.add_argument('--sp-admm-elem-per-row', type=int, default=1)
    admm_args.add_argument('--sp-admm-tile', type=str, default=None,
                        help="in the form of (x,y) e.g. (256,256) \
                            x is the number of rows in a tile, -1 means all rows \
                            y is the number of cols in a tile, -1 means all cols")

    # the following is for M:N pruning sparsity type
    admm_args.add_argument('--sp-admm-select-number', type=int, default=4)
    admm_args.add_argument('--sp-admm-pattern-row-sub', type=int, default=1)
    admm_args.add_argument('--sp-admm-pattern-col-sub', type=int, default=4)
    admm_args.add_argument('--sp-admm-data-format', type=str, default=None, help="define sp_admm_format: [NHWC,NCHW], ")
    admm_args.add_argument('--sp-admm-do-not-permute-conv', default=False, action='store_true', help="Do not permute conv filters ")



    # output compressed format
    admm_args.add_argument('--sp-gs-output-v', type=str, default=None, help="output compressed format of a gs pattern ")
    admm_args.add_argument('--sp-gs-output-ptr', type=str, default=None, help="output compressed format of a gs pattern ")


class ADMM(PruneBase):
    def __init__(self, args, model, logger=None, initialize=True):
        super(ADMM, self).__init__(args, model, logger)
        # this is to keep in CPU
        self.ADMM_U = {}
        self.ADMM_Z = {}
        # this is the identical copy in GPU. We do this separation
        # because in some cases in GPU run out of space if modified
        # directly
        self.ADMM_U_GPU = {}
        self.ADMM_Z_GPU = {}
        self.rhos = {}
        self.rho = None
        self.args = args


        assert args.sp_config_file is not None, "Config file must be specified for ADMM"
        self.logger.info("Initializing ADMM pruning algorithm")

        if self.args.sp_admm_update_epoch is not None:
            self.update_epoch = self.args.sp_admm_update_epoch
        elif 'admm_update_epoch' in self.configs:
            self.update_epoch = self.configs["admm_update_epoch"]
        else:
            self.update_epoch = None
        if self.args.sp_admm_update_batch is not None:
            self.update_batch = self.args.sp_admm_update_batch
        elif 'admm_update_batch' in self.configs:
            self.update_batch = self.configs["admm_update_batch"]
        else:
            self.update_batch = None

        assert (self.update_epoch is None and self.update_batch is not None) or \
               (self.update_epoch is not None and self.update_batch is None)

        assert self.prune_ratios is not None
        if 'rho' in self.configs:
            self.rho = self.configs['rho']
        else:
            assert self.args.sp_admm_rho is not None
            self.rho = self.args.sp_admm_rho
        self.logger.info("ADMM rho is set to {}".format(str(self.rho)))

        if self.args.sp_load_prune_params is not None:
            self.prune_load_params()
        elif initialize:
            self.init()

        if self.args.admm_debug:
            self.admm_debug = True
        else:
            self.admm_debug = False

    def init(self):
        first = True
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:
                continue
            self.rhos[name] = self.rho
            prune_ratio = self.prune_ratios[name]


            self.logger.info("ADMM initialzing {}".format(name))
            updated_Z = self.prune_weight(name, W, prune_ratio, first)  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            #print("Done")


            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()
            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U[name] = torch.zeros(W.shape).detach().cpu().float()
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)

        if (self.args.output_compressed_format) and (self.args.sp_gs_output_v is not None) and (self.args.sp_gs_output_ptr is not None):
            print("Compressed format output done!")
            exit()

    def prune_harden(self, option=None):
        if self.args.sp_no_harden:
            self.logger.info("Not hardening the matrix")
            return
        super(ADMM, self).prune_harden()


        if self.args.sp_global_weight_sparsity > 0:
            update_prune_ratio(self.args, self.model, self.prune_ratios, self.args.sp_global_weight_sparsity, self.args.sp_prune_threshold)

        for key in self.prune_ratios:
            print("prune_ratios[{}]:{}".format(key,self.prune_ratios[key]))

        #self.logger.info("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
        print("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
        first = True
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:  # ignore layers that do not have rho
                continue
            cuda_pruned_weights = None
            prune_ratio = self.prune_ratios[name]
            if option == None:
                cuda_pruned_weights = self.prune_weight(name, W, prune_ratio, first)  # get sparse model in cuda
                first = False

            elif option == "random":
                _, cuda_pruned_weights = random_pruning(self.args, W, prune_ratio)

            elif option == "l1":
                _, cuda_pruned_weights = L1_pruning(self.args, W, prune_ratio)
            else:
                raise Exception("not implmented yet")
            W.data = cuda_pruned_weights.cuda().type(W.dtype)  # replace the data field in variable

            if self.args.sp_admm_sparsity_type == "block":
                block = eval(self.args.sp_admm_block)
                if block[1] == -1: # row pruning, need to delete corresponding bias
                    bias_layer = name.replace(".weight", ".bias")
                    with torch.no_grad():
                        bias = self.model.state_dict()[bias_layer]
                        bias_mask = torch.sum(W, 1)
                        bias_mask[bias_mask != 0] = 1
                        bias.mul_(bias_mask)
            elif self.args.sp_admm_sparsity_type == "filter" or self.args.sp_admm_sparsity_type == "filter_CSS":
                if not "downsample" in name:
                    bn_weight_name = name.replace("conv", "bn")
                    bn_bias_name = bn_weight_name.replace("weight", "bias")
                else:
                    bn_weight_name = name.replace("downsample.0", "downsample.1")
                    bn_bias_name = bn_weight_name.replace("weight", "bias")

                print("removing bn {}, {}".format(bn_weight_name, bn_bias_name))
                # bias_layer_name = name.replace(".weight", ".bias")

                with torch.no_grad():
                    bn_weight = self.model.state_dict()[bn_weight_name]
                    bn_bias = self.model.state_dict()[bn_bias_name]
                    # bias = self.model.state_dict()[bias_layer_name]

                    mask = torch.sum(torch.abs(W), (1,2,3))
                    mask[mask != 0] = 1
                    bn_weight.mul_(mask)
                    bn_bias.mul_(mask)
                    # bias.data.mul_(mask)


            non_zeros = W.detach().cpu().numpy() != 0
            non_zeros = non_zeros.astype(np.float32)
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            print("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
            #self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))

    # this is expected to be in the very beginning of the epoch
    def prune_update(self, epoch=0, batch_idx=0):
        if ((self.update_epoch is not None) and ((epoch == 0) or \
            (epoch % self.update_epoch != 0))) or \
            ((self.update_batch is not None) and ((batch_idx == 0) or  \
             (batch_idx % self.update_batch != 0))) :
            return

        super(ADMM, self).prune_update(epoch, batch_idx)
        # this is to avoid the bug that GPU memory overflow
        for key in self.ADMM_Z:
            del self.ADMM_Z_GPU[key]
        for key in self.ADMM_U:
            del self.ADMM_U_GPU[key]
        first = True
        for i, (name, W) in enumerate(self.model.named_parameters()):
            if name not in self.prune_ratios:
                continue
            Z_prev = None
            W_CPU = W.detach().cpu().float()

            admm_z = W_CPU + self.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            updated_Z = self.prune_weight(name, admm_z,
                                          self.prune_ratios[name], first)  # equivalent to Euclidean Projection
            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()

            self.ADMM_U[name] = (W_CPU - self.ADMM_Z[name] + self.ADMM_U[name]).float()  # U(k+1) = W(k+1) - Z(k+1) +U(k)

            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)

    def prune_update_combined_loss(self, ce_loss):
        admm_loss = {}
        for i, (name, W) in enumerate(self.model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in self.prune_ratios:
                continue
            if self.prune_ratios[name] == 0.0:
                continue
            admm_loss[name] = (0.5 * self.rhos[name] * \
                (torch.norm(W.float() - self.ADMM_Z_GPU[name].float() +
                self.ADMM_U_GPU[name].float(), p=2) ** 2)).float()

        total_admm_loss = 0
        for k, v in admm_loss.items():
            total_admm_loss += v
        mixed_loss = total_admm_loss + ce_loss

        if self.admm_debug:
            ce_loss_np = ce_loss.data.cpu().numpy()
            _admm_loss_np = total_admm_loss.data.cpu().numpy()
            print("ce_loss:{}, admm_loss:{}, mixed_loss:{}".format(
                  ce_loss_np, _admm_loss_np, mixed_loss.data.cpu().numpy()))


        return ce_loss, admm_loss, mixed_loss

    def prune_update_loss(self, ce_loss):
        _, _, combined_loss = self.prune_update_combined_loss(ce_loss)
        return combined_loss

    def prune_load_params(self):
        variables = self._prune_load_params()
        if variables == None:
            return
        self.logger.info("Loading ADMM variables")
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:
                continue
            if self.prune_ratios[name] == 0.0:
                continue
            cname = self._canonical_name(name)
            n = name if name in variables["U"] else cname
            if n not in variables["U"]:
                self.logger.warning("Param {} cannot be found in saved param file".format(n))
            self.ADMM_U[name] = variables["U"][n]
            self.ADMM_Z[name] = variables["Z"][n]
            self.rhos[name] = variables["rhos"][n]
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)
            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)

    def prune_store_params(self):
        if not self.args.sp_store_prune_params:
            return
        self.logger.info("Storing ADMM variables")
        variables = {
            "U": self.ADMM_U,
            "Z": self.ADMM_Z,
            "rhos": self.rhos,
        }
        self._prune_store_params(variables)

    def prune_weight(self, name, weight, prune_ratio, first):
        if prune_ratio == 0.0:
            return weight
        # if pruning too many items, just prune everything
        if prune_ratio >= 0.999:
            return weight * 0.0
        if self.args.sp_admm_sparsity_type == "irregular_global":
            res = self.weight_pruning_irregular_global(weight,
                                                       prune_ratio, first)
        else:
            if (self.args.sp_gs_output_v is not None) and (self.args.sp_gs_output_ptr is not None):
                print("Start to output layer {}".format(name))

            sp_admm_sparsity_type_copy = copy.copy(self.args.sp_admm_sparsity_type)
            sparsity_type_list = (self.args.sp_admm_sparsity_type).split("+")
            if len(sparsity_type_list) != 1: #multiple sparsity type
                print(sparsity_type_list)
                for i in range(len(sparsity_type_list)):
                    sparsity_type = sparsity_type_list[i]
                    print("* sparsity type {} is {}".format(i, sparsity_type))
                    self.args.sp_admm_sparsity_type = sparsity_type
                    _, weight =  weight_pruning(self.args, self.configs, name, weight, prune_ratio)
                    self.args.sp_admm_sparsity_type = sp_admm_sparsity_type_copy
                    print(np.sum(weight.detach().cpu().numpy() != 0))
                return weight.to(weight.device).type(weight.dtype)
            else:
                _, res = weight_pruning(self.args, self.configs, name, weight, prune_ratio)


        return res.to(weight.device).type(weight.dtype)

    def weight_pruning_irregular_global(self, weight, prune_ratio, first):
        with torch.no_grad():
            if first:
                self.irregular_global_blob = None
                total_size = 0
                for i, (name, W) in enumerate(self.model.named_parameters()):
                    if name not in self.prune_ratios:
                        continue
                    if self.prune_ratios[name] == 0.0:
                        continue
                    total_size += W.numel()
                to_prune = torch.zeros(total_size)
                index_ = 0
                for (name, W) in self.model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    if self.prune_ratios[name] == 0.0:
                        continue
                    size = W.numel()
                    to_prune[index_:(index_+size)] = W.view(-1).abs().clone()
                    index_ += size
                sorted_to_prune, _ = torch.sort(to_prune)
                self.irregular_global_blob = sorted_to_prune

            total_size = self.irregular_global_blob.numel()
            thre_index = int(total_size * prune_ratio)
            global_th = self.irregular_global_blob[thre_index]
            above_threshold = (weight.detach().cpu().float().abs() >
                global_th).to(weight.device).type(weight.dtype)
            weight = (weight * above_threshold).type(weight.dtype)
            return weight



def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sp_admm_sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def update_prune_ratio(args, model, prune_ratios, global_sparsity, sp_prune_threshold=-1.0):
    # prune layers in prune_ratios only if the sparsity of this layer is < prune_sparsity_threshold
    if sp_prune_threshold > 0:
        for name, W in (model.named_parameters()):
            if (canonical_name(name) in prune_ratios.keys()) or (name in prune_ratios.keys()):
                sp_W = 1 - float(np.sum(W.detach().cpu().numpy() != 0))/W.data.numel()
                print(name, sp_W)
                if sp_W > sp_prune_threshold:
                    prune_ratios.pop(name, None)
        #print(prune_ratios)
        #exit()


    total_size = 0
    for name, W in (model.named_parameters()):

        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        total_size += W.data.numel()
    to_prune = np.zeros(total_size)
    index = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        to_prune[index:(index+size)] = W.data.clone().cpu().view(-1).abs().numpy()
        index += size
    #sorted_to_prune = np.sort(to_prune)
    threshold = np.percentile(to_prune, global_sparsity*100)

    # update prune_ratios key-value pairs
    total_zeros = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) \
                and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        np_W_abs = W.detach().cpu().abs().numpy()
        new_prune_ratio = float(np.sum(np_W_abs < threshold))/size

        total_zeros += float(np.sum(np_W_abs < threshold))

        prune_ratios[name] = new_prune_ratio

    print("Updated prune_ratios:")
    for key in prune_ratios:
        print("prune_ratios[{}]:{}".format(key,prune_ratios[key]))
    total_sparsity = total_zeros / total_size
    print("Total sparsity:{}".format(total_sparsity))


    return prune_ratios

def weight_growing(args, name, pruned_weight_np, lower_bound_value, upper_bound_value, update_init_method, mask_fixed_params=None):
    shape = None
    weight1d = None

    if mask_fixed_params is not None:
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    if upper_bound_value == 0:
        print("==> GROW: {}: to DENSE despite the sparsity type is \n".format(name))
        np_updated_mask = np.ones_like(pruned_weight_np, dtype=np.float32)
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if upper_bound_value == lower_bound_value:
        print("==> GROW: {}: no grow, keep the mask and do finetune \n".format(name))
        non_zeros_updated = pruned_weight_np != 0
        non_zeros_updated = non_zeros_updated.astype(np.float32)
        np_updated_mask = non_zeros_updated
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if (args.sp_admm_sparsity_type == "irregular"):
        # randomly select and set zero weights to non-zero to restore sparsity
        non_zeros_prune = pruned_weight_np != 0

        shape = pruned_weight_np.shape
        weight1d = pruned_weight_np.reshape(1, -1)[0]
        zeros_indices = np.where(weight1d == 0)[0]
        if args.sp_global_magnitude:
            num_added_zeros = int((lower_bound_value - upper_bound_value) * np.size(weight1d))
        else:
            num_added_zeros = int(np.size(zeros_indices) - upper_bound_value * np.size(weight1d))
        num_added_zeros = num_added_zeros if num_added_zeros < np.size(zeros_indices) else np.size(zeros_indices)
        num_added_zeros = num_added_zeros if num_added_zeros > 0 else 0
        target_sparsity = 1 - (np.count_nonzero(non_zeros_prune) + num_added_zeros) * 1.0 / np.size(pruned_weight_np)
        indices = np.random.choice(zeros_indices,
                                   num_added_zeros,
                                   replace=False)
        print("==> CALCULATE: all zeros: {}, need grow {} zeros, selected zeros: {} ".format(len(zeros_indices),
                                                                                             num_added_zeros,
                                                                                             len(indices)))

        # initialize selected weights
        if update_init_method == "weight":
            current_nozero = weight1d[np.nonzero(weight1d)]
            current_mean = np.mean(current_nozero)
            current_std = np.std(current_nozero)
            weight1d[indices] = np.random.normal(loc=current_mean, scale=current_std, size=np.size(indices))

            weight = weight1d.reshape(shape)

            print("==> double check sparsity after updating mask...")
            non_zeros_updated = weight != 0
            non_zeros_updated = non_zeros_updated.astype(np.float32)
            num_nonzeros_updated = np.count_nonzero(non_zeros_updated)
            sparsity_updated = 1 - (num_nonzeros_updated * 1.0) / total_num
            print(("{}: {}, {}, {}\n".format(name, str(num_nonzeros_updated), str(total_num), str(sparsity_updated))))

            # update mask
            # zero_mask = torch.from_numpy(non_zeros_updated).cuda()
            np_updated_zero_one_mask = non_zeros_updated

            # write updated weights back to model
            model.state_dict()[name].data.copy_(torch.from_numpy(weight))
        elif update_init_method == "zero":
            # set selected weights to -1 to get corrrect updated masks
            weight1d[indices] = -1
            weight = weight1d.reshape(shape)
            non_zeros_updated = weight != 0
            non_zeros_updated = non_zeros_updated.astype(np.float32)
            print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, target_sparsity))

            # update mask
            # zero_mask = torch.from_numpy(non_zeros_updated).cuda()
            np_updated_zero_one_mask = non_zeros_updated

            # assign 0 to -1 weight
            weight1d[indices] = 0
            weight = weight1d.reshape(shape)

            # write updated weights back to model
            # self.model.state_dict()[name].data.copy_(torch.from_numpy(weight))
        elif update_init_method == "kaiming":
            assert (False)

        np_updated_mask = np_updated_zero_one_mask
        updated_mask = torch.from_numpy(np_updated_mask).cuda()

        return updated_mask

    elif args.sp_admm_sparsity_type == "N:M-prune-pattern+block":
        shape = pruned_weight_np.shape
        weight2d = copy.copy(pruned_weight_np)
        if len(shape) == 2:
            # assume it is MN format
            pass
        elif len(shape) == 4:
            # assume it is CoCIKhKw format
            # first serialize KhKw to one dimension
            co, ci, kh, kw = weight2d.shape
            weight2d = weight2d.reshape([co, ci, kh * kw])
            # convert from CoCiKhKw to CoKhKwCi format
            weight2d = np.moveaxis(weight2d, 1, -1)
            # merge Ci, Kh, Kw dimension
            weight2d = weight2d.reshape([co, ci * kh * kw])
        elif len(shape) == 3:
            co, ci, kl = weight2d.shape
            weight2d = np.moveaxis(weight2d, 1, -1)
            weight2d = weight2d.reshape([co, ci * kl])
        else:
            assert False, "matrix dim = {}, not equal to 2 (MM), 3 (1d Conv), or 4 (2d Conv)!".format(len(shape))

        assert len(weight2d.shape) == 2, "Now only support 2d matrices"

        block = args.sp_admm_block
        block = eval(block)
        # print(block[0],block[1])
        # exit()
        row_pad_num = (block[0] - weight2d.shape[0] % block[0]) % block[0]
        col_pad_num = (block[1] - weight2d.shape[1] % block[1]) % block[1]
        new_weight2d = np.zeros((weight2d.shape[0] + row_pad_num, weight2d.shape[1] + col_pad_num))
        new_weight2d[:weight2d.shape[0], :weight2d.shape[1]] = weight2d
        new_weight2d = np.sqrt(new_weight2d * new_weight2d)

        '''
        np.set_printoptions(precision=2)
        np.set_printoptions(threshold=sys.maxsize)
        if args.local_rank==0:
            print(weight.shape)
            print(new_weight2d.shape)
            print(new_weight2d[:24,:24])
        '''

        if block[0] == -1:
            block_l = list(block)
            block_l[0] = new_weight2d.shape[0]
            block = tuple(block_l)
        elif block[1] == -1:
            block_l = list(block)
            block_l[1] = new_weight2d.shape[1]
            block = tuple(block_l)
        block_size = block[0]*block[1]
        partitioned_weight2d = view_as_windows(new_weight2d, block, step=block)
        sum2d = np.sum(partitioned_weight2d, axis=(2, 3))
        sum2d = (sum2d != 0).astype(np.float32)
        sum2d_shape = sum2d.shape
        sum1d = sum2d.reshape(1, -1)[0]
        zeros_indices = np.where(sum1d == 0)[0]
        num_added_zeros = int(np.size(zeros_indices) - upper_bound_value * np.size(sum1d))
        indices = np.random.choice(zeros_indices,
                                   num_added_zeros,
                                   replace=False)
        print("==> CALCULATE: all zeros: {}, need grow {} zeros, selected zeros: {} ".format(len(zeros_indices) * block_size,
                                                                                             num_added_zeros * block_size,
                                                                                             len(indices) * block_size))

        sum1d[indices] = 1
        sum2d = sum1d.reshape(sum2d_shape)
        sum2d = sum2d * 1.0
        growing_sparsity = np.sum(sum2d == 0) / np.size(sum2d)
        print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, growing_sparsity))

        mask2d = np.kron(sum2d, np.ones(block))
        mask2d = mask2d[:weight2d.shape[0], :weight2d.shape[1]]
        if len(shape) == 2:
            # assume it is MN format
            growing_mask = mask2d
        elif len(shape) == 4:
            # assume it is CoCIKhKw format
            co, ci, kh, kw = pruned_weight_np.shape
            # first separate out Ci, Kh, Kw dimensions
            mask2d = mask2d.reshape([co, kh, kw, ci])
            # convert from CoKhKwCi to CoCiKhKw format
            growing_mask = np.moveaxis(mask2d, -1, 1)
        elif len(shape) == 3:
            co, ci, kl = pruned_weight_np.shape
            mask2d = mask2d.reshape([co, kl, ci])
            growing_mask = np.moveaxis(mask2d, -1, 1)

        assert pruned_weight_np.shape == growing_mask.shape, "Mask shape not equal to weights shape!"

        growing_mask = torch.from_numpy(growing_mask).cuda()
        return growing_mask


    elif (args.sp_admm_sparsity_type == "4:2-H-V-balanced"):
        # data format transponse
        block_row_size = 4
        block_col_size = 4
        if (args.sp_admm_data_format == "NCHW" and len(pruned_weight_np.shape)==4) : pruned_weight_np = np.transpose(pruned_weight_np, (0, 3, 1, 2))  # NHWC to NCHW
        if (args.sp_admm_data_format == "NHWC" and len(pruned_weight_np.shape)==4) : pruned_weight_np = np.transpose(pruned_weight_np, (0, 2, 3, 1))  # NCHW to NHWC
        weight_abs      = np.abs(pruned_weight_np)
        shape           = pruned_weight_np.shape
        weight2d        = pruned_weight_np.reshape(shape[0], -1)
        weight2d_abs    = np.abs(weight2d)
        shape2d         = weight2d.shape
        # args.sp_admm_pattern_col_sub * args.sp_admm_pattern_row_sub : select_number sparsity pattern
        pattern_col_num         = shape2d[1] // block_col_size
        pattern_col_remainder   = shape2d[1] %  block_row_size
        pattern_row_num         = shape2d[0] // block_col_size
        pattern_row_remainder   = shape2d[0] %  block_row_size
        weight2d_abs_pad        = np.pad(weight2d_abs, ((0,0 if pattern_row_remainder==0 else block_row_size-pattern_row_remainder),
                                                        (0,0 if pattern_col_remainder==0 else block_col_size-pattern_col_remainder)),
                                                        'constant', constant_values=0)
        weight2d_pad = np.pad(weight2d, ((0,0 if pattern_row_remainder==0 else block_row_size-pattern_row_remainder),
                                         (0,0 if pattern_col_remainder==0 else block_col_size-pattern_col_remainder)),
                                         'constant', constant_values=0)
        shape2d_pad = weight2d_abs_pad.shape
        pattern_col_pad_num = shape2d_pad[1] // block_col_size
        pattern_row_pad_num = shape2d_pad[0] // block_row_size
        #print(weight2d_abs_pad[:10,:10])
        block_mask_rxc = np.random.rand(pattern_row_pad_num,pattern_col_pad_num) < (0.5 - upper_bound_value) # with prob. threshold, the mask is 1 so we grow that block to dense
        block_mask_all = np.kron(block_mask_rxc, np.ones([block_row_size,block_col_size]))

        weight_mask = weight2d_abs_pad!=0
        growing_mask = (weight_mask + block_mask_all)>0

        if (args.sp_admm_data_format == "NCHW" and len(pruned_weight_np.shape)==4) : pruned_weight_np = np.transpose(growing_mask, (0, 2, 3, 1))  # NCHW to NHWC
        if (args.sp_admm_data_format == "NHWC" and len(pruned_weight_np.shape)==4) : pruned_weight_np = np.transpose(growing_mask, (0, 3, 1, 2))  # NHWC to NCHW

        growing_sparsity = np.sum(growing_mask == 0) / np.size(growing_mask)
        print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, growing_sparsity))

        growing_mask = torch.from_numpy(growing_mask).cuda()
        return growing_mask



def four_two_pruning(args, weight, pattern='hvb', percent=0.5):
    if args.sp_admm_sparsity_type == "4:2-2:1":
        print ("using 4:2-2:1")
        candidate=[[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],]]
        candidate_4_2 = np.array(candidate)
        candidate_4_2_flatten = candidate_4_2.reshape(36,16)
    else:
        print ("using 4:2-H-V-balanced")
        candidate=[[[1, 1, 0, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 0, 1, 1, ],],[[1, 1, 0, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 0, 1, 1, ],],[[1, 1, 0, 0, ],[1, 0, 1, 0, ],[0, 0, 1, 1, ],[0, 1, 0, 1, ],],[[1, 1, 0, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 0, 1, 1, ],],[[1, 1, 0, 0, ],[1, 0, 0, 1, ],[0, 0, 1, 1, ],[0, 1, 1, 0, ],],[[1, 1, 0, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 0, 1, 1, ],],[[1, 1, 0, 0, ],[0, 1, 1, 0, ],[0, 0, 1, 1, ],[1, 0, 0, 1, ],],[[1, 1, 0, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 0, 1, 1, ],],[[1, 1, 0, 0, ],[0, 1, 0, 1, ],[0, 0, 1, 1, ],[1, 0, 1, 0, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[1, 0, 1, 0, ],[1, 1, 0, 0, ],[0, 1, 0, 1, ],[0, 0, 1, 1, ],],[[1, 0, 1, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 1, 0, 1, ],],[[1, 0, 1, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 1, 0, 1, ],],[[1, 0, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 1, 0, 1, ],],[[1, 0, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 0, 1, ],[0, 1, 1, 0, ],],[[1, 0, 1, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 0, 1, ],],[[1, 0, 1, 0, ],[0, 1, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 0, 1, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[1, 0, 1, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 1, 0, 1, ],],[[1, 0, 1, 0, ],[0, 0, 1, 1, ],[0, 1, 0, 1, ],[1, 1, 0, 0, ],],[[1, 0, 0, 1, ],[1, 1, 0, 0, ],[0, 1, 1, 0, ],[0, 0, 1, 1, ],],[[1, 0, 0, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],[0, 1, 1, 0, ],],[[1, 0, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 1, 0, ],[0, 1, 0, 1, ],],[[1, 0, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],[0, 1, 1, 0, ],],[[1, 0, 0, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 1, 1, 0, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[1, 0, 0, 1, ],[0, 1, 1, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[1, 0, 0, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 1, 0, ],],[[1, 0, 0, 1, ],[0, 1, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 1, 0, ],],[[1, 0, 0, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 1, 1, 0, ],],[[1, 0, 0, 1, ],[0, 0, 1, 1, ],[0, 1, 1, 0, ],[1, 1, 0, 0, ],],[[0, 1, 1, 0, ],[1, 1, 0, 0, ],[1, 0, 0, 1, ],[0, 0, 1, 1, ],],[[0, 1, 1, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 0, 0, 1, ],],[[0, 1, 1, 0, ],[1, 0, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 0, 1, ],],[[0, 1, 1, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 0, 1, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[0, 1, 1, 0, ],[1, 0, 0, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[0, 1, 1, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 0, 0, 1, ],],[[0, 1, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 0, 0, 1, ],],[[0, 1, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 0, 1, ],[1, 0, 1, 0, ],],[[0, 1, 1, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 0, 0, 1, ],],[[0, 1, 1, 0, ],[0, 0, 1, 1, ],[1, 0, 0, 1, ],[1, 1, 0, 0, ],],[[0, 1, 0, 1, ],[1, 1, 0, 0, ],[1, 0, 1, 0, ],[0, 0, 1, 1, ],],[[0, 1, 0, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 0, 1, 0, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[0, 1, 0, 1, ],[1, 0, 1, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[0, 1, 0, 1, ],[1, 0, 0, 1, ],[1, 0, 1, 0, ],[0, 1, 1, 0, ],],[[0, 1, 0, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 1, 0, ],],[[0, 1, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 1, 0, ],[1, 0, 0, 1, ],],[[0, 1, 0, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 0, 1, 0, ],],[[0, 1, 0, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 0, 1, 0, ],],[[0, 1, 0, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 0, 1, 0, ],],[[0, 1, 0, 1, ],[0, 0, 1, 1, ],[1, 0, 1, 0, ],[1, 1, 0, 0, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],],[[0, 0, 1, 1, ],[1, 1, 0, 0, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],],[[0, 0, 1, 1, ],[1, 0, 1, 0, ],[1, 1, 0, 0, ],[0, 1, 0, 1, ],],[[0, 0, 1, 1, ],[1, 0, 1, 0, ],[0, 1, 0, 1, ],[1, 1, 0, 0, ],],[[0, 0, 1, 1, ],[1, 0, 0, 1, ],[1, 1, 0, 0, ],[0, 1, 1, 0, ],],[[0, 0, 1, 1, ],[1, 0, 0, 1, ],[0, 1, 1, 0, ],[1, 1, 0, 0, ],],[[0, 0, 1, 1, ],[0, 1, 1, 0, ],[1, 1, 0, 0, ],[1, 0, 0, 1, ],],[[0, 0, 1, 1, ],[0, 1, 1, 0, ],[1, 0, 0, 1, ],[1, 1, 0, 0, ],],[[0, 0, 1, 1, ],[0, 1, 0, 1, ],[1, 1, 0, 0, ],[1, 0, 1, 0, ],],[[0, 0, 1, 1, ],[0, 1, 0, 1, ],[1, 0, 1, 0, ],[1, 1, 0, 0, ],],[[0, 0, 1, 1, ],[0, 0, 1, 1, ],[1, 1, 0, 0, ],[1, 1, 0, 0, ],]]
        candidate_4_2 = np.array(candidate)
        candidate_4_2_flatten = candidate_4_2.reshape(90,16)

    # Assume pytorch use
    # OIHW to OHWI
    shape_before = weight.shape
    if len(weight.shape) == 4 and not args.sp_admm_do_not_permute_conv:
        weight = np.transpose(weight, (0, 2, 3, 1))
    #print("after reshape:", weight[:4,0,0,:4])


    weight_abs      = np.abs(weight)
    shape           = weight.shape
    weight2d        = weight.reshape(shape[0], -1)
    weight2d_abs    = np.abs(weight2d)
    shape2d         = weight2d.shape
    # args.sp_admm_pattern_col_sub * args.sp_admm_pattern_row_sub : select_number sparsity pattern
    pattern_col_num         = shape2d[1] // 4
    pattern_col_remainder   = shape2d[1] %  4
    pattern_row_num         = shape2d[0] // 4
    pattern_row_remainder   = shape2d[0] %  4

    weight2d_abs_pad        = np.pad(weight2d_abs, ((0,0 if pattern_row_remainder==0 else 4-pattern_row_remainder),
                                                    (0,0 if pattern_col_remainder==0 else 4-pattern_col_remainder)),
                                                    'constant', constant_values=0)
    weight2d_pad = np.pad(weight2d, ((0,0 if pattern_row_remainder==0 else 4-pattern_row_remainder),
                                     (0,0 if pattern_col_remainder==0 else 4-pattern_col_remainder)),
                                     'constant', constant_values=0)
    shape2d_pad = weight2d_abs_pad.shape
    pattern_col_pad_num = shape2d_pad[1] // 4
    pattern_row_pad_num = shape2d_pad[0] // 4
    #print(weight2d_abs_pad[:10,:10])
    def check_valid(mat):
        assert mat.shape == (4,4), 'Matrix not 4x4!'
        row_sum = np.sum(mat!=0,axis=0)
        col_sum = np.sum(mat!=0,axis=1)
        #print(mat, row_sum, col_sum)
        if row_sum[0]==2 and row_sum[1]==2 and row_sum[2]==2 and row_sum[3]==2 and col_sum[0]==2 and col_sum[1]==2 and col_sum[2]==2 and col_sum[3]==2:
            return True
        else:
            return False

    block=(4,4)
    partitioned_weight2d = view_as_windows(weight2d_abs_pad, block, step=block)
    partitioned_weight2d_flatten = partitioned_weight2d.reshape(partitioned_weight2d.shape[0], partitioned_weight2d.shape[1], -1)
    candidate_sum = np.inner(candidate_4_2_flatten,partitioned_weight2d_flatten)
    max_idx_array = np.argmax(candidate_sum, axis=0)

    blocked_mask2d = partitioned_weight2d * 0

    #print(max_idx_array, max_idx_array.shape)
    final_mask = 0 * weight2d_abs_pad
    for i in range(pattern_row_pad_num):
        for j in range(pattern_col_pad_num):
            final_mask[i*4:(i+1)*4,j*4:(j+1)*4] = candidate_4_2[max_idx_array[i][j]]
    weight2d_pad *= final_mask

    weight2d = weight2d_pad
    for i in range(4-pattern_row_remainder):
        if pattern_row_remainder!=0: weight2d = np.delete(weight2d, shape2d_pad[0] -1 - i, axis=0)
    for i in range(4-pattern_col_remainder):
        if pattern_col_remainder!=0: weight2d = np.delete(weight2d, shape2d_pad[1] -1 - i, axis=1)
    weight = weight2d.reshape(shape)
    # Assume pytorch use OIHW
    # OHWI bach to OIHW
    if len(weight.shape) == 4 and not args.sp_admm_do_not_permute_conv:
        weight = np.transpose(weight, (0, 3, 1, 2))
    shape_after = weight.shape

    non_zeros = weight != 0
    non_zeros = non_zeros.astype(np.float32)
    num_nonzeros = np.count_nonzero(non_zeros)
    total_num = non_zeros.size
    sparsity = 1 - (num_nonzeros * 1.0) / total_num
    print ("num_nonzeros ", num_nonzeros, "total_num ", total_num, "sparsity", sparsity)

    return non_zeros, weight



def block_pruning(args, weight, percent, return_block_sums=False):
    print("using block pruning...")
    shape = weight.shape
    weight2d = copy.copy(weight)

    if len(shape) == 2:
        # assume it is MN format
        pass
    elif len(shape) == 4:
        # assume it is CoCIKhKw format
        # first serialize KhKw to one dimension
        co, ci, kh, kw = weight2d.shape
        weight2d = weight2d.reshape([co, ci, kh * kw])
        # convert from CoCiKhKw to CoKhKwCi format
        weight2d = np.moveaxis(weight2d, 1, -1)
        # merge Ci, Kh, Kw dimension
        weight2d = weight2d.reshape([co, ci * kh * kw])
    elif len(shape) == 3:
        co, ci, kl = weight2d.shape
        weight2d = np.moveaxis(weight2d, 1, -1)
        weight2d = weight2d.reshape([co, ci * kl])
    else:
        assert False, "matrix dim = {}, not equal to 2 (MM), 3 (1d Conv), or 4 (2d Conv)!".format(len(shape))


    assert len(weight2d.shape) == 2, "Now only support 2d matrices"

    block = args.sp_admm_block
    block = eval(block)
    row_pad_num = (block[0] - weight2d.shape[0] % block[0]) % block[0]
    col_pad_num = (block[1] - weight2d.shape[1] % block[1]) % block[1]
    new_weight2d = np.zeros((weight2d.shape[0]+row_pad_num, weight2d.shape[1]+col_pad_num))
    new_weight2d[:weight2d.shape[0], :weight2d.shape[1]] = weight2d
    new_weight2d = np.sqrt(new_weight2d * new_weight2d)

    if block[0] == -1:
        block_l = list(block)
        block_l[0] = new_weight2d.shape[0]
        block = tuple(block_l)
    elif block[1] == -1:
        block_l = list(block)
        block_l[1] = new_weight2d.shape[1]
        block = tuple(block_l)
    partitioned_weight2d = view_as_windows(new_weight2d, block, step=block)
    sum2d = np.sum(partitioned_weight2d, axis=(2,3))
    percentile = np.percentile(sum2d, percent)
    above_threshold = (sum2d > percentile) + 0.0

    # output block index information for CSR
    if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
        mask = copy.copy(above_threshold)
        num_cmf = [0]
        col_indices = []
        for row in mask:
            #print(row)
            num_this_row = np.sum(row)
            col_idx = np.array(np.where(row>0)).flatten()
            for c in col_idx:
                col_indices.append(c)

            num_cmf.append(num_this_row + num_cmf[-1])

        with open(args.sp_gs_output_ptr+"_"+name+".txt", 'a') as f:
            f.write("{} {}\n".format(weight2d.shape[0],weight2d.shape[1]))
            f.write("{} {}\n".format(block[0],block[1]))
            for c in col_indices:
                f.write("{} ".format(int(c)))
            f.write("\n")
            for cmf in num_cmf:
                f.write("{} ".format(int(cmf)))

    mask2d = np.kron(above_threshold, np.ones(block))
    mask2d = mask2d[:weight2d.shape[0],:weight2d.shape[1]]

    if len(shape) == 2:
        # assume it is MN format
        pass
    elif len(shape) == 4:
        # assume it is CoCIKhKw format
        co, ci, kh, kw = weight.shape
        # first separate out Ci, Kh, Kw dimensions
        mask2d = mask2d.reshape([co, kh, kw, ci])
        # convert from CoKhKwCi to CoCiKhKw format
        mask2d = np.moveaxis(mask2d, -1, 1)
    elif len(shape) == 3:
        co, ci, kl = weight.shape
        mask2d = mask2d.reshape([co, kl, ci])
        mask2d = np.moveaxis(mask2d, -1, 1)

    assert weight.shape == mask2d.shape, "Mask shape not equal to weights shape!"
    masked_w = weight * mask2d

    if return_block_sums:
        assert len(shape) == 2, "return windowned block masks, now only support weight as 2D matrices!"
        return mask2d, masked_w, sum2d
    else:
        return mask2d, masked_w

def irregular_pruning(args, weight, percent):
    weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
    percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
    under_threshold = weight_temp < percentile
    above_threshold = weight_temp > percentile
    above_threshold = above_threshold.astype(
        np.float32)  # has to convert bool to float32 for numpy-tensor conversion
    # weight[under_threshold] = 0
    ww = weight * above_threshold
    return above_threshold, ww


def block_interleaved_4_2_pruning(args, weight, percent):
    mask2d, masked_w, sum2d =  block_pruning(args, weight, percent, True)
    block_size = eval(args.sp_admm_block)
    percentile = np.percentile(sum2d, percent)
    above_threshold = (sum2d > percentile) + 0.0

    non_zero_block_per_row = np.sum(above_threshold,axis=1) #number of element > threshold per row

    interleaved_block_multiplier = 16

    non_zero_block_per_row_aligned = (non_zero_block_per_row + interleaved_block_multiplier/2)//interleaved_block_multiplier * interleaved_block_multiplier

    percentile_per_row = (1-(non_zero_block_per_row_aligned+0.0)/sum2d.shape[1]) * 100

    threshold_each_row = []
    for i in range(sum2d.shape[0]):
        threshold_each_row.append(np.percentile(sum2d[i],percentile_per_row[i]))
    threshold_each_row = np.array(threshold_each_row)

    threshold_all = np.repeat(np.expand_dims(threshold_each_row, axis=1),sum2d.shape[1],axis=1)

    above_threshold = (sum2d > threshold_all) + 0.0
    # back to weight mask
    mask2d = np.kron(above_threshold, np.ones(block_size))

    weight_blocked_pruned = weight * mask2d

    cnt = 0
    buffer_i = np.zeros(4)
    buffer_j = np.zeros(4)
    abs_ww = []
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            if mask2d[i][j] == 1:
                abs_ww.append((abs(weight_blocked_pruned[i][j]),i,j))
                cnt += 1
            if cnt == 4:
                abs_ww.sort()
                weight_blocked_pruned[abs_ww[0][1]][abs_ww[0][2]] = 0
                weight_blocked_pruned[abs_ww[1][1]][abs_ww[1][2]] = 0
                cnt = 0
                abs_ww = []
    mask = (weight_blocked_pruned != 0) + 0.0
    return mask, weight_blocked_pruned

def weight_pruning(args, configs, name, w, prune_ratio, mask_fixed_params=None):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """
    torch_weight = w
    weight = w.detach().clone().cpu().numpy()  # convert cpu tensor to numpy
    if mask_fixed_params is not None:
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    percent = prune_ratio * 100

    if (args.sp_admm_sparsity_type == "irregular"):
        mask, masked_w = irregular_pruning(args, weight, percent)
        return torch.from_numpy(mask), torch.from_numpy(masked_w)

        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        # weight[under_threshold] = 0
        ww = weight * above_threshold
        return torch.from_numpy(above_threshold), torch.from_numpy(ww)
    elif (args.sp_admm_sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "channel"):
        shape = weight.shape
        print("channel pruning...", weight.shape)
        weight3d = weight.reshape(shape[0], shape[1], -1)
        channel_l2_norm = LA.norm(weight3d, 2, axis=(0,2))
        percentile = np.percentile(channel_l2_norm, percent)
        under_threshold = channel_l2_norm <= percentile
        above_threshold = channel_l2_norm > percentile
        weight3d[:,under_threshold,:] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(weight3d.shape, dtype=np.float32)
        for i in range(weight3d.shape[1]):
            expand_above_threshold[:, i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "vector"):
        shape = weight.shape
        weight2d = weight.reshape(-1, 16)
        shape2d = weight2d.shape
        percentile = np.percentile(weight2d, percent,axis = 1)
        percentile = np.reshape(percentile,[-1, 1])
        percentile = np.repeat(percentile,16,axis=1)
        under_threshold = weight2d <= percentile
        above_threshold = weight2d > percentile
        weight2d[under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
#        expand_above_threshold = np.zeros(weight3d.shape, dtype=np.float32)
#        expand_above_threshold = above_threshold+0
        weight = weight.reshape(shape)
        above_threshold = above_threshold.reshape(shape)
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif args.sp_admm_sparsity_type == "hybrid_block":
        print("using hybrid block pruning...")
        shape = weight.shape
        weight2d = copy.copy(weight)
        if len(shape) == 2:
            pass
        else:
            assert False, "matrix dim = {}, not equal to 2 (MM), 3 (1d Conv), or 4 (2d Conv)!".format(len(shape))

        # original block
        block = args.sp_admm_block
        block = eval(block)
        block_tr = np.array(block)[::-1]
        #print(block)
        #print(block_tr)


        row_pad_num = (block[0] - weight2d.shape[0] % block[0]) % block[0]
        col_pad_num = (block[1] - weight2d.shape[1] % block[1]) % block[1]
        new_weight2d = np.zeros((weight2d.shape[0]+row_pad_num, weight2d.shape[1]+col_pad_num))
        new_weight2d[:weight2d.shape[0], :weight2d.shape[1]] = weight2d
        new_weight2d = np.sqrt(new_weight2d * new_weight2d)

        partitioned_weight2d_1 = view_as_windows(new_weight2d, block, step=block)
        partitioned_weight2d_2 = view_as_windows(new_weight2d, block_tr, step=block_tr)

        #print(partitioned_weight2d_1.shape)
        #print(partitioned_weight2d_2.shape)

        sum2d_1 = np.sum(partitioned_weight2d_1, axis=(2,3))
        sum2d_2 = np.sum(partitioned_weight2d_2, axis=(2,3))
        #print(sum2d_1.shape)
        #print(sum2d_2.shape)
        sum2d_1_flatten = sum2d_1.flatten()
        sum2d_2_flatten = sum2d_2.flatten()
        #print(sum2d_1_flatten)
        #print(sum2d_2_flatten)
        #print(sum2d_1_flatten[:8])
        #print(sum2d_1[0,:8])
        sum2d_combined = np.vstack((sum2d_1_flatten,sum2d_2_flatten))
        #print(sum2d_combined.shape)
        #print(sum2d_combined[:2,:8])
        new_percent = 100-(100-percent)/2
        percentile = np.percentile(sum2d_combined, new_percent)
        above_threshold = (sum2d_combined > percentile) + 0.0

        mask2d_1 = np.kron(np.reshape(above_threshold[0],sum2d_1.shape), np.ones(block))
        mask2d_2 = np.kron(np.reshape(above_threshold[1],sum2d_2.shape), np.ones(block_tr))
        #print(mask2d_1.shape)
        #print(mask2d_2.shape)
        #print(mask2d_1[24:24+16,24:24+16])
        #print(mask2d_2[24:24+16,24:24+16])
        mask2d = ((mask2d_1 + mask2d_2) > 0) + 0.0
        mask2d = mask2d[:weight2d.shape[0],:weight2d.shape[1]]
        #print(mask2d[24:24+16,24:24+16])
        assert weight.shape == mask2d.shape, "Mask shape not equal to weights shape!"
        masked_w = weight * mask2d


        return torch.from_numpy(mask2d), torch.from_numpy(masked_w)

    elif args.sp_admm_sparsity_type == "block_same_pattern":
        pass
        print("using block with the same pattern pruning...")
        shape = weight.shape
        weight2d = copy.copy(weight)
        assert len(weight2d.shape) == 2, "Now only support 2d matrices"

        block = args.sp_admm_block
        block = eval(block)

        row_pad_num = (block[0] - weight2d.shape[0] % block[0]) % block[0]
        col_pad_num = (block[1] - weight2d.shape[1] % block[1]) % block[1]
        new_weight2d = np.zeros((weight2d.shape[0]+row_pad_num, weight2d.shape[1]+col_pad_num))
        new_weight2d[:weight2d.shape[0], :weight2d.shape[1]] = weight2d
        new_weight2d = np.sqrt(new_weight2d * new_weight2d)

        partitioned_weight2d = view_as_windows(new_weight2d, block, step=block)
        #print(partitioned_weight2d.shape)

        sum2d = np.sum(partitioned_weight2d, axis=(0,1))
        #print(sum2d)
        percentile = np.percentile(sum2d, percent)
        #print(percent, percentile)
        above_threshold = (sum2d > percentile) + 0.0
        #print(above_threshold)

        mask2d = np.tile(above_threshold, partitioned_weight2d.shape[:2])
        mask2d = mask2d[:weight2d.shape[0],:weight2d.shape[1]]

        assert weight.shape == mask2d.shape, "Mask shape not equal to weights shape!"
        masked_w = weight * mask2d

        return torch.from_numpy(mask2d), torch.from_numpy(masked_w)

    elif args.sp_admm_sparsity_type == "block_union_irregular":
        (block_sparsity, irregular_sparsity) = eval(args.sp_block_irregular_sparsity)
        block_percent = block_sparsity * 100
        irregular_percent = irregular_sparsity * 100
        print("using block + irregular pruning...")
        mask_block, _ = block_pruning(args, weight, block_percent)
        mask_irregular, _ = irregular_pruning(args, weight, irregular_percent)

        mask = ((mask_block + mask_irregular) > 0) + 0.0
        mask_w = weight * mask
        return torch.from_numpy(mask), torch.from_numpy(mask_w)

    elif args.sp_admm_sparsity_type == "block":
        mask2d, masked_w =  block_pruning(args, weight, percent)
        return torch.from_numpy(mask2d), torch.from_numpy(masked_w)

    elif args.sp_admm_sparsity_type == "block_row_permuted_interleaved_4_2" or args.sp_admm_sparsity_type == "block_row_permuted":
        permute_percentile = 100 - (100 - percent) * 2
        print("Permute percentile, percent:",permute_percentile, percent)
        w_tmp = np.abs(weight)
        irr_percentile = np.percentile(w_tmp, permute_percentile)
        irr_mask = ((w_tmp > irr_percentile) + 0.0).astype(int)

        from k_means_constrained import KMeansConstrained
        mat_rows = weight.shape[0]
        block_size = eval(args.sp_admm_block)
        b_size = block_size[0]
        n_clusters = mat_rows/b_size

        kmeans = KMeansConstrained(n_clusters=int(n_clusters), size_min=b_size,size_max=b_size,random_state=0)
        kmeans.fit_predict(irr_mask)

        #np.set_printoptions(linewidth=200,threshold=sys.maxsize)
        permute_idx = np.zeros(len(kmeans.labels_))
        inverse_permuate_idx = permute_idx * 0
        for c_ind in range(int(n_clusters)):
            idxs = np.where(kmeans.labels_==int(c_ind))
            permute_idx[c_ind*b_size:(c_ind+1)*b_size] = np.array(idxs)
        permute_idx = permute_idx.astype(int)
        #print(permute_idx)
        for i in range(len(permute_idx)):
            inverse_permuate_idx[permute_idx[i]] = i
        inverse_permuate_idx = inverse_permuate_idx.astype(int)

        # row permutations
        weight_permuted = weight * 0
        weight_permuted_back = weight * 0
        for i in range(weight_permuted.shape[0]):
            weight_permuted[i] = weight[permute_idx[i]]

        if args.sp_admm_sparsity_type == "block_row_permuted_interleaved_4_2":
            mask2d, masked_w = block_interleaved_4_2_pruning(args, weight_permuted, percent)
        else:
            mask2d, masked_w = block_pruning(args, weight_permuted, percent)

        for i in range(weight_permuted_back.shape[0]):
            weight_permuted_back[i] = masked_w[inverse_permuate_idx[i]]

        #diff = np.sum(abs(weight_permuted_back - weight))

        #print("permute vector:", permute_idx)
        #print("Diff:", diff)

        mask2d = (weight_permuted_back != 0) + 0.0
        #input("?")
        return torch.from_numpy(mask2d), torch.from_numpy(weight_permuted_back)


    elif args.sp_admm_sparsity_type == "interleaved_block_intersect_four_two":
        mask2d, masked_w = block_interleaved_4_2_pruning(args, weight, percent)
        return torch.from_numpy(mask2d), torch.from_numpy(masked_w)



    elif args.sp_admm_sparsity_type == "filter_CSS":
        def CSS(weight, k):
            '''
            k: pruning rate, i.e. select (1-k)*C columns
            '''
            shape = weight.shape
            #print(shape)
            #input("?")
            X = np.reshape(weight, (weight.shape[0], -1))
            X = np.transpose(X)
            if X.shape[0] >= X.shape[1]:
                _, _, V = np.linalg.svd(X)
                Vk = V[:,:int((1-k)*X.shape[1])]
                lvs = np.linalg.norm(Vk, axis=1)
                #lvs = lvs.cpu().numpy()
                return lvs
            else:
                #weight_copy = copy.copy(np.abs(weight))
                norm = np.sum(weight, axis=(1,2,3))
                return norm
        assert len(weight.shape) == 4, f"Filter shape should be a 4-dim tensor, not {weight.shape}"

        num_output_filter = weight.shape[1]
        score = CSS(weight, prune_ratio)
        score_thr = np.percentile(score, percent)
        under_thr = score < score_thr
        above_thr = score > score_thr
        weight[under_thr,:,:,:] = 0
        mask = weight > 0
        #print(np.sum(weight,axis=(1,2,3)))
        #print(score, score_thr)
        #input("?")
        return torch.from_numpy(mask).cuda(), torch.from_numpy(weight).cuda()

    elif (args.sp_admm_sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "gather_scatter"):
        shape = weight.shape
        weight_abs = np.abs(weight)
        if ("block_shape" in configs) and (name in configs["block_shape"]):
            block = configs["block_shape"][name]
        else:
            block = args.sp_admm_block
        block_shape=eval(block)
        num_buckets=args.sp_admm_buckets_num
        # bucket_axis=args.sp_admm_bucket_axis
        elem_per_row = args.sp_admm_elem_per_row
        tile = args.sp_admm_tile

        # get the structure
        shape_len = len(shape)
        start_idx = shape_len - len(block_shape)
        new_shape = list(shape[:start_idx])
        new_reduce_shape_idx = []
        for i in range(start_idx, shape_len, 1):
            idx = i - start_idx
#            print("shape:",shape)
#            print("block_shape:",block_shape)
#            print("i:", i)
#            print("idx:", idx)
            assert shape[i] % block_shape[idx] == 0
            new_shape.append(shape[i] // block_shape[idx])
            new_reduce_shape_idx.append(len(new_shape))
            new_shape.append(block_shape[idx])
        new_weight_abs = weight_abs.reshape(new_shape)
        sum_array = np.sum(new_weight_abs, tuple(new_reduce_shape_idx))
        sum_array_shape = sum_array.shape
#        print("new_reduce_shape_idx:", new_reduce_shape_idx)
#        print("new_shape:", new_shape)
#        print("new_weight_abs:", new_weight_abs.shape)
#        print("sum_array_shape:", sum_array_shape)

        orig_sum_array = sum_array
        # We need to handle CoCIKhKw or MN formats
        if len(sum_array_shape) == 2:
            # assume it is MN format
            pass
        elif len(sum_array_shape) == 4:
            # assume it is CoCIKhKw format
            # first serialize KhKw to one dimension
            co, ci, kh, kw = sum_array_shape
            sum_array = sum_array.reshape([co, ci, kh * kw])
            # convert from CoCiKhKw to CoKhKwCi format
            sum_array = np.moveaxis(sum_array, 1, -1)
            # merge Ci, Kh, Kw dimension
            sum_array = sum_array.reshape([co, ci * kh * kw])
        elif len(sum_array_shape) == 3:
            co, ci, kl = sum_array_shape
            sum_array = np.moveaxis(sum_array, 1, -1)
            sum_array = sum_array.reshape([co, ci * kl])

        else:
            assert False
        '''
        if num_buckets > 1 and bucket_axis >= 0:
            ndims = len(sum_array.shape)
            if ndims > bucket_axis + 1:
                start = bucket_axis + 1
                if sum_array.shape[bucket_axis+1] != 1:
                    bucket_is_last_axis = False
                    start = bucket_axis + 2
                if start < ndims:
                    reduce_axis = [x for x in range(start, ndims, 1)]
                    sum_array = np.sum(sum_array, tuple(reduce_axis))
        '''
#                print("reduce_axis:", reduce_axis)

        threshold = np.percentile(sum_array, percent)
        mask = sum_array > threshold
        mask = mask * 1 # convert to int
#        print("mask_shape1:",mask.shape)
        # import pdb; pdb.set_trace()
        if num_buckets > 1:
            mask = np.zeros(sum_array.shape)
            update_bucket_mask(sum_array, mask, threshold,
                               num_buckets,
                               elem_per_row, args.sp_admm_tile, args, name)
#        print("mask_shape2:",mask.shape)
        # rewrite the shape back
        if len(sum_array_shape) == 2:
            # assume it is MN format
            pass
        elif len(sum_array_shape) == 4:
            # assume it is CoCIKhKw format
            co, ci, kh, kw = sum_array_shape
            # first separate out Ci, Kh, Kw dimensions
            mask = mask.reshape([co, kh, kw, ci])
            # convert from CoKhKwCi to CoCiKhKw format
            mask = np.moveaxis(mask, -1, 1)
        elif len(sum_array_shape) == 3:
            co, ci, kl = sum_array_shape
            mask = mask.reshape([co, kl, ci])
            mask = np.moveaxis(mask, -1, 1)

        else:
            assert False
        mask = np.reshape(mask, sum_array_shape)
#        print(mask.shape)

        if len(block_shape) > 1 and (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
            #np.set_printoptions(threshold=sys.maxsize)
            #print(mask[:25,:25],mask.shape)
            #print(float(np.sum(mask))/np.size(mask))
            num_cmf = [0]
            col_indices = []
            for row in mask:
                #print(row)
                num_this_row = np.sum(row)
                col_idx = np.array(np.where(row>0)).flatten()
                for c in col_idx:
                    col_indices.append(c)

                #print(col_indices)
                num_cmf.append(num_this_row + num_cmf[-1])
                #print(num_cmf)
                #input("?")

            with open(args.sp_gs_output_ptr+"_"+name+".txt", 'a') as f:
                f.write("{} {}\n".format(weight.shape[0],weight.shape[1]))
                f.write("{} {}\n".format(block_shape[0],block_shape[1]))
                for c in col_indices:
                    f.write("{} ".format(c))
                f.write("\n")
                for cmf in num_cmf:
                    f.write("{} ".format(cmf))
            #input("?")


        for i in range(len(block_shape) - 1, -1, -1):
            idx = len(mask.shape) - len(block_shape) + i
            if block_shape[i] == 1:
                continue
            mask = np.repeat(mask, block_shape[i], idx)



        torch_mask = (torch.from_numpy(mask)).type(torch_weight.dtype).to(torch_weight.device)
        new_weight = torch_weight * torch_mask
#        for i in range(weight.shape[0]):
#            print(name, i ,((np.shape(np.where(weight[i,:]!=0)))[1]))
#        print(np.remainder(np.array(np.where(weight[0,:]!=0)),16))
        return threshold, new_weight.cpu()

    elif (args.sp_admm_sparsity_type == "aligned_pattern"):
        print("aligned pattern pruning...", weight.shape)
        shape = weight.shape

        """pattern WM mobile (4 pattern)"""
        pattern1 = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]

        pattern2 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
        pattern3 = [[0, 0], [0, 1], [0, 2], [2, 1], [2, 2]]
        pattern4 = [[0, 2], [1, 0], [1, 2], [2, 0], [2, 2]]
        pattern5 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
        pattern6 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 2]]
        pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern8 = [[0, 0], [1, 0], [1, 2], [2, 0], [2, 2]]
        pattern9 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]

        pattern10 = [[0, 1], [0, 2], [1, 2], [2, 0], [2, 2]]
        pattern11 = [[0, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
        pattern12 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 1]]
        pattern13 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
        pattern14 = [[0, 0], [0, 1], [1, 0], [2, 0], [2, 2]]
        pattern15 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
        pattern16 = [[0, 0], [0, 2], [1, 2], [2, 1], [2, 2]]
        pattern17 = [[0, 2], [1, 0], [2, 0], [2, 1], [2, 2]]

        pattern18 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern19 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]
        pattern20 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
        pattern21 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8,
                         9: pattern9,
                         10: pattern10,
                         11: pattern11,
                         12: pattern12,
                         13: pattern13,
                         14: pattern14,
                         15: pattern15,
                         16: pattern16,
                         17: pattern17,
                         18: pattern18,
                         19: pattern19,
                         20: pattern20,
                         21: pattern21
                         }

        # weight3d = weight.reshape(weight.shape[0], weight.shape[1], -1)
        for i in range(shape[1]):  # loop channel
            current_channel = weight[:, i, :, :].copy()
            temp_dict = {}  # store each pattern's norm value on the same weight kernel
            for key, pattern in patterns_dict.items():
                temp_channel = current_channel.copy()
                for j in range(temp_channel.shape[0]):  # loop every kernel in a channel
                    for index in pattern:
                        temp_channel[j, :][index[0], index[1]] = 0
                current_norm = LA.norm(temp_channel)
                temp_dict[key] = current_norm
            best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
            for index in patterns_dict[best_pattern]:
                weight[:, i, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "pattern"):
        print("pattern pruning...", weight.shape)
        shape = weight.shape

        """pattern Tsinghua hardware (5 pattern)"""
        # pattern1 = [[0, 0], [0, 2], [2, 0], [2, 2]]
        # pattern2 = [[0, 0], [0, 1], [2, 1], [2, 2]]
        # pattern3 = [[0, 0], [0, 1], [2, 0], [2, 1]]
        # pattern4 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        #
        # pattern5 = [[0, 2], [1, 0], [1, 2], [2, 0]]
        # pattern6 = [[0, 0], [1, 0], [1, 2], [2, 2]]
        # pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1]]
        # pattern8 = [[0, 1], [0, 2], [2, 1], [2, 2]]
        #
        # pattern9 = [[1, 0], [1, 2], [2, 0], [2, 2]]
        # pattern10 = [[0, 0], [0, 2], [1, 0], [1, 2]]
        # pattern11 = [[1, 1], [1, 2], [2, 1], [2, 2]]
        # pattern12 = [[1, 0], [1, 1], [2, 0], [2, 1]]
        # pattern13 = [[0, 1], [0, 2], [1, 1], [1, 2]]
        #
        # patterns_dict = {1 : pattern1,
        #                  2 : pattern2,
        #                  3 : pattern3,
        #                  4 : pattern4,
        #                  5 : pattern5,
        #                  6 : pattern6,
        #                  7 : pattern7,
        #                  8 : pattern8,
        #                  9 : pattern9,
        #                  10 : pattern10,
        #                  11 : pattern11,
        #                  12 : pattern12,
        #                  13 : pattern13
        #                  }

        """pattern WM mobile (4 pattern)"""
        pattern1 = [[0, 0], [0, 2], [1, 1], [2, 0], [2, 2]]

        pattern2 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 0]]
        pattern3 = [[0, 0], [0, 1], [0, 2], [2, 1], [2, 2]]
        pattern4 = [[0, 2], [1, 0], [1, 2], [2, 0], [2, 2]]
        pattern5 = [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2]]
        pattern6 = [[0, 0], [0, 2], [1, 0], [1, 2], [2, 2]]
        pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern8 = [[0, 0], [1, 0], [1, 2], [2, 0], [2, 2]]
        pattern9 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1]]

        pattern10 = [[0, 1], [0, 2], [1, 2], [2, 0], [2, 2]]
        pattern11 = [[0, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
        pattern12 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 1]]
        pattern13 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 2]]
        pattern14 = [[0, 0], [0, 1], [1, 0], [2, 0], [2, 2]]
        pattern15 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]
        pattern16 = [[0, 0], [0, 2], [1, 2], [2, 1], [2, 2]]
        pattern17 = [[0, 2], [1, 0], [2, 0], [2, 1], [2, 2]]

        pattern18 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern19 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]
        pattern20 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]
        pattern21 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8,
                         9: pattern9,
                         10: pattern10,
                         11: pattern11,
                         12: pattern12,
                         13: pattern13,
                         14: pattern14,
                         15: pattern15,
                         16: pattern16,
                         17: pattern17,
                         18: pattern18,
                         19: pattern19,
                         20: pattern20,
                         21: pattern21
                         }

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                temp_dict = {} # store each pattern's norm value on the same weight kernel
                for key, pattern in patterns_dict.items():
                    temp_kernel = current_kernel.copy()
                    for index in pattern:
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                for index in patterns_dict[best_pattern]:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "filter_balance"):
        print("pruning filter with balanced outputs")

        kth_smallest = prune_ratio  # the percent from script is used to represent k-th smallest l2-norm kernel will be pruned in each filter
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)
        for i in range(shape[0]):
            kernel_l2norm_list = LA.norm(weight3d[i,:,:], 2, axis=1)
            partial_sorted_index = np.argpartition(kernel_l2norm_list, kth_smallest)  # list of all indices, but partially sorted
            kth_smallest_index = partial_sorted_index[:kth_smallest]  # indices of k-th smallest l2-norm
            for idx in kth_smallest_index:
                weight3d[i, idx, :] = 0
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "two_filter_balance_1"):
        pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
        pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
        pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
        pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

        pattern5 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
        pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
        pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53


        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8
                         }

        print("pruning two filter with balanced outputs -- step 1: group aligned pattern")

        shape = weight.shape
        numFilter = shape[0]

        weight2d = weight.reshape(shape[0], -1)
        filter_L2 = LA.norm(weight2d, 2, axis=1)
        weight3d = weight.reshape(shape[0], shape[1], -1)

        filter_index_dict = {}
        for index, l2_item in enumerate(filter_L2):
            filter_index_dict[index] = l2_item
        filter_index_dict = sorted(filter_index_dict.items(), key=lambda k: [k[1], k[0]])
        filter_index_dict = collections.OrderedDict(filter_index_dict)
        sorted_filter_index = list(filter_index_dict.keys())

        if os.path.exists("./{}.pkl".format(name)):
            os.remove("./{}.pkl".format(name))
        afile = open(r"./{}.pkl".format(name), 'wb')
        pickle.dump(sorted_filter_index, afile)
        afile.close()

        for i, (filter_idx, _) in enumerate(filter_index_dict.items()):
            if i % 4 == 0:
                first_idx = filter_idx
                second_idx = list(filter_index_dict.keys())[i + 1]
                third_idx = list(filter_index_dict.keys())[i + 2]
                forth_idx = list(filter_index_dict.keys())[i + 3]
                temp = np.array([weight3d[first_idx, :, :], weight3d[second_idx, :, :], weight3d[third_idx, :, :], weight3d[forth_idx, :, :]])

                """add aligned pattern prune for this current pair before aligned connectivity prune"""
                temp = temp.reshape([temp.shape[0], temp.shape[1], 3, 3])
                for k in range(temp.shape[1]):  # loop channel
                    current_channel = temp[:, k, :, :].copy()
                    temp_dict = {}  # store each pattern's norm value on the same weight kernel
                    for key, pattern in patterns_dict.items():
                        temp_channel = current_channel.copy()
                        for j in range(temp_channel.shape[0]):  # loop every kernel in a channel
                            for index in pattern:
                                temp_channel[j, :][index[0], index[1]] = 0
                        current_norm = LA.norm(temp_channel)
                        temp_dict[key] = current_norm
                    best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                    for index in patterns_dict[best_pattern]:
                        temp[:, k, index[0], index[1]] = 0
                temp = temp.reshape([temp.shape[0], temp.shape[1], -1])

                """aligned connectivity prune"""
                if percent == 0:
                    weight3d[first_idx] = temp[0]
                    weight3d[second_idx] = temp[1]
                    weight3d[third_idx] = temp[2]
                    weight3d[forth_idx] = temp[3]
                    continue
                channel_l2_norm = LA.norm(temp, 2, axis=(0, 2))
                if i <= numFilter / 4:
                    percentile = np.percentile(channel_l2_norm, percent / 1)
                elif numFilter / 4 < i <= numFilter / 2:
                    percentile = np.percentile(channel_l2_norm, percent / 1)
                elif numFilter / 2 < i <= numFilter:
                    percentile = np.percentile(channel_l2_norm, percent)
                under_threshold = channel_l2_norm <= percentile
                above_threshold = channel_l2_norm > percentile
                temp[:, under_threshold, :] = 0

                weight3d[first_idx] = temp[0]
                weight3d[second_idx] = temp[1]
                weight3d[third_idx] = temp[2]
                weight3d[forth_idx] = temp[3]

        weight = weight3d.reshape(shape)

        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "two_filter_balance_2"):
        print("pruning two filter with balanced outputs -- step 2: group aligned connectivity")

        file = open(r"./{}.pkl".format(name), 'rb')
        sorted_filter_index = pickle.load(file)
        file.close()

        shape = weight.shape
        numFilter = shape[0]
        weight3d = weight.reshape(shape[0], shape[1], -1)

        for i, filter_idx in enumerate(sorted_filter_index):
            if i % 4 == 0:
                first_idx = filter_idx
                second_idx = list(sorted_filter_index)[i + 1]
                third_idx = list(sorted_filter_index)[i + 2]
                forth_idx = list(sorted_filter_index)[i + 3]
                temp = np.array([weight3d[first_idx, :, :], weight3d[second_idx, :, :], weight3d[third_idx, :, :], weight3d[forth_idx, :, :]])

                if percent == 0:
                    weight3d[first_idx] = temp[0]
                    weight3d[second_idx] = temp[1]
                    weight3d[third_idx] = temp[2]
                    weight3d[forth_idx] = temp[3]
                    continue
                channel_l2_norm = LA.norm(temp, 2, axis=(0, 2))
                if i <= numFilter / 4:
                    percentile = np.percentile(channel_l2_norm, percent / 1.2)
                elif numFilter / 4 < i <= numFilter / 2:
                    percentile = np.percentile(channel_l2_norm, percent / 1.1)
                elif numFilter / 2 < i <= numFilter:
                    percentile = np.percentile(channel_l2_norm, percent)
                under_threshold = channel_l2_norm <= percentile
                above_threshold = channel_l2_norm > percentile
                temp[:, under_threshold, :] = 0
                weight3d[first_idx] = temp[0]
                weight3d[second_idx] = temp[1]
                weight3d[third_idx] = temp[2]
                weight3d[forth_idx] = temp[3]

        weight = weight3d.reshape(shape)

        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sp_admm_sparsity_type == "4:2-H-V-balanced" or args.sp_admm_sparsity_type == "4:2-2:1"):
        mask2d, masked_w = four_two_pruning(args, weight)
        return torch.from_numpy(mask2d), torch.from_numpy(masked_w)

    elif args.sp_admm_sparsity_type == "block_intersect_four_two":
        mask_4_2, _ = four_two_pruning(args, weight)
        mask_block, _ = block_pruning(args, weight, percent)

        mask = (mask_4_2 * mask_block > 0) + 0.0
        masked_w = weight * mask
        return torch.from_numpy(mask), torch.from_numpy(masked_w)

    elif args.sp_admm_sparsity_type == "block_intersect_four_two_union_irregular":

        (block_sparsity, irregular_sparsity) = eval(args.sp_block_irregular_sparsity)
        block_percent = block_sparsity * 100
        irregular_percent = irregular_sparsity * 100
        print("using block intersection 4:2 union irregular pruning...")
        mask_block, _ = block_pruning(args, weight, block_percent)
        mask_irregular, _ = irregular_pruning(args, weight, irregular_percent)
        mask_4_2, _ = four_two_pruning(args, weight)

        mask = ((mask_block * mask_4_2 + mask_irregular)  > 0) + 0.0
        mask_w = weight * mask
        return torch.from_numpy(mask), torch.from_numpy(mask_w)

    elif (args.sp_admm_sparsity_type == "N:M-prune-pattern"):
        print ("using N:M-prune-pattern")


        # Assume pytorch use
        # OIHW to OHWI
        if len(weight.shape) == 4 and not args.sp_admm_do_not_permute_conv:
            weight = np.transpose(weight, (0, 2, 3, 1))


        weight_abs      = np.abs(weight)
        shape           = weight.shape
        weight2d        = weight.reshape(shape[0], -1)
        weight2d_abs    = np.abs(weight2d)
        shape2d         = weight2d.shape

        # args.sp_admm_pattern_col_sub * args.sp_admm_pattern_row_sub : select_number sparsity pattern
        pattern_col_num         = shape2d[1] // args.sp_admm_pattern_col_sub
        pattern_col_remainder   = shape2d[1] %  args.sp_admm_pattern_col_sub
        pattern_row_num         = shape2d[0] // args.sp_admm_pattern_row_sub
        pattern_row_remainder   = shape2d[0] %  args.sp_admm_pattern_row_sub
        weight2d_abs_pad        = np.pad(weight2d_abs, ((0,0 if pattern_row_remainder==0 else args.sp_admm_pattern_row_sub-pattern_row_remainder),
                                                        (0,0 if pattern_col_remainder==0 else args.sp_admm_pattern_col_sub-pattern_col_remainder)),
                                                        'constant', constant_values=0)
        weight2d_pad = np.pad(weight2d, ((0,0 if pattern_row_remainder==0 else args.sp_admm_pattern_row_sub-pattern_row_remainder),
                                         (0,0 if pattern_col_remainder==0 else args.sp_admm_pattern_col_sub-pattern_col_remainder)),
                                         'constant', constant_values=0)
        shape2d_pad = weight2d_abs_pad.shape
        pattern_col_pad_num = shape2d_pad[1] // args.sp_admm_pattern_col_sub
        pattern_row_pad_num = shape2d_pad[0] // args.sp_admm_pattern_row_sub
        for i in range(pattern_row_pad_num):
            for j in range(pattern_col_pad_num):
                weight_pattern = weight2d_abs_pad[i*args.sp_admm_pattern_row_sub:(i+1)*args.sp_admm_pattern_row_sub,
                                                  j*args.sp_admm_pattern_col_sub:(j+1)*args.sp_admm_pattern_col_sub].flatten()

                tmp_list = copy.deepcopy(weight_pattern.tolist())
                tmp_list.sort()
                min_num_index_list = [weight_pattern.tolist().index(one) for one in tmp_list[:args.sp_admm_pattern_row_sub*args.sp_admm_pattern_col_sub-args.sp_admm_select_number]]

                min_num = [one for one in tmp_list[:args.sp_admm_pattern_row_sub * args.sp_admm_pattern_col_sub - args.sp_admm_select_number]]
                for r in range(len(list(list_duplicates(min_num)))):
                    for p in range(len(list(list_duplicates(weight_pattern)))):
                        if list(list_duplicates(weight_pattern))[p][0] == list(list_duplicates(min_num))[r][0]:
                            min_num_index_list = np.append(min_num_index_list,
                                            list(list_duplicates(weight_pattern))[p][1][1:len(list(list_duplicates(min_num))[r][1])])


                for flatten_index in list(min_num_index_list):
                    row_index_num = flatten_index // args.sp_admm_pattern_col_sub
                    col_index_num = flatten_index % args.sp_admm_pattern_col_sub
                    weight2d_pad[i*args.sp_admm_pattern_row_sub+row_index_num, j*args.sp_admm_pattern_col_sub+col_index_num] = 0
        weight2d = weight2d_pad
        for i in range(args.sp_admm_pattern_row_sub-pattern_row_remainder):
            if pattern_row_remainder!=0: weight2d = np.delete(weight2d, shape2d_pad[0] -1 - i, axis=0)
        for i in range(args.sp_admm_pattern_col_sub-pattern_col_remainder):
            if pattern_col_remainder!=0: weight2d = np.delete(weight2d, shape2d_pad[1] -1 - i, axis=1)
        weight = weight2d.reshape(shape)


        # Assume pytorch use OIHW
        # OHWI bach to OIHW
        if len(weight.shape) == 4 and not args.sp_admm_do_not_permute_conv:
            weight = np.transpose(weight, (0, 3, 1, 2))

        # np.set_printoptions(threshold=100, edgeitems=8)
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print ("num_nonzeros ", num_nonzeros, "total_num ", total_num, "sparsity", sparsity)
        return torch.from_numpy(non_zeros), torch.from_numpy(weight)

    raise SyntaxError("Unknown sparsity type: {}".format(args.sp_admm_sparsity_type))

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)

#    for (name, W) in model.named_parameters():
#        ADMM.ADMM_U[name] = torch.zeros(W.shape).cuda()




ttt = 0




def admm_multi_rho_scheduler(ADMM, name):
    """
    It works better to make rho monotonically increasing
    we increase it by 1.9x every admm epoch
    After 10 admm updates, the rho will be 0.91

    """

    ADMM.rhos[name] *= 2


def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every args.sp_admm_update_epoch/3 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default
    admm epoch is 9)

    """
    admm_epoch = args.sp_admm_update_epoch
    lr = None

    if (epoch) % admm_epoch == 0:
        lr = args.sp_admm_lr
    else:
        admm_epoch_offset = (epoch) % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = args.sp_admm_lr * (0.1 ** (admm_epoch_offset // admm_step))

    #print(admm_epoch, args.sp_admm_lr, (epoch) % admm_epoch, lr)
    #input('?')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def threshold_mask(param, threshold):
    """Create a threshold mask for the provided parameter tensor using
    magnitude thresholding.
    Arguments:
        param: a parameter tensor which should be pruned.
        threshold: the pruning threshold.
    Returns:
        prune_mask: The pruning mask.
    """
    return torch.gt(torch.abs(param), threshold).type(param.type())


def zero_masking(args, config, model):
    masks = {}
    for name, W in model.named_parameters():  ## no gradient for weights that are already zero (for progressive pruning and sequential pruning)
        if name in config.prune_ratios:
            w_temp = W.cpu().detach().numpy()
            indices = (w_temp != 0)
            indices = indices.astype(np.float32)
            masks[name] = torch.from_numpy(indices).cuda()
    config.zero_masks = masks

'''
def masking(args, config, model):
    masks = {}
    for name, W in model.named_parameters():
        if name in config.prune_ratios:
            above_threshold, pruned_weight = weight_pruning(args, W, config.prune_ratios[name])
            W.data = pruned_weight
            masks[name] = above_threshold

    config.masks = masks
'''


def generate_mask(model):
    masks = {}
    for name, W in (model.named_parameters()):
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask
    return masks


def update_subarray_bucket_mask(sum_array, mask, threshold, num_buckets,
                                start_idx, end_idx, elem_per_row, args=None, name=None):
    def get_value(v):
        return v[2]

    def get_max_pack(sorted_bucket_values, num_rows, elem_per_row):
        one_pack = [[None for i in range(elem_per_row)] for j in range(num_rows)]
        col_indices = [0] * num_rows
        occupied_bucket = [None] * num_buckets
        num = 0
        while num < num_rows * elem_per_row:
            max_row = 0
            max_value = None
            max_idx = -1
            for j in range(num_rows):
                if col_indices[j] >= len(one_pack[j]):
                    continue
                for k in range(len(sorted_bucket_values[j])-1, -1, -1):
                    item = sorted_bucket_values[j][k]
                    col = item[1]
                    bucket = col % num_buckets
                    if occupied_bucket[bucket] == None:
                        if max_value is None or (max_value[2] < item[2]):
                            max_row = j
                            max_value = item
                            max_idx = k
                        break
            if max_value is None:
                break
            sorted_bucket_values[max_row].pop(max_idx)
            assert one_pack[max_row][col_indices[max_row]] == None
            one_pack[max_row][col_indices[max_row]] = max_value
            col_indices[max_row] = col_indices[max_row] + 1
            num = num + 1
            bucket = max_value[1] % num_buckets
            assert occupied_bucket[bucket] == None
            occupied_bucket[bucket] = True
        return one_pack

    assert(len(sum_array.shape) == 2)

    num_items = np.count_nonzero(np.abs(sum_array[start_idx:end_idx]) > threshold) # np.sum(ref_mask[start_idx: end_idx])
    # whether to add one more elements or reduce one element?
    if num_items // num_buckets > 8:
        num_items = ((num_items + num_buckets // 2) // num_buckets) * num_buckets

    sorted_bucket_values = []
    num_rows = end_idx - start_idx

    assert num_rows * elem_per_row <= num_buckets
    sorted_bucket_values = [[]] * num_rows

    for i in range(num_rows):
        bucket_values = []
        for j in range(sum_array.shape[1]):
            bucket_values.append((start_idx + i, j, sum_array[start_idx + i][j]))
        sorted_bucket_values[i] = sorted(bucket_values, key=get_value)

    num_packs = 0
    while num_items > 0:
        one_pack = get_max_pack(sorted_bucket_values, num_rows, elem_per_row)
        #print(one_pack)
        if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
            one_line_v = []
            one_line_col_idx = []
            for one_row in one_pack:
                for z in one_row:
                    one_line_v.append(z[2])
                    one_line_col_idx.append(z[1])
            while len(one_line_v) < args.sp_admm_buckets_num:
                one_line_v.append(0.0)
                one_line_col_idx.append(sum_array.shape[1])
                print("Some bucket not filled!")

        num_packs += 1
        #print(one_line_v)
        #print(one_line_col_idx)
        #print(num_packs)

        if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
            with open(args.sp_gs_output_v+"_"+name+".txt", 'a') as f:
                for v in one_line_v:
                    f.write("{} ".format(v))
                f.write("\n")
                for col in one_line_col_idx:
                    f.write("{} ".format(col))
                f.write("\n")

        #input("?")
        num_items = num_items - num_buckets
        for i in range(num_rows):
            for j in range(elem_per_row):
                elem = one_pack[i][j]
                if elem != None:
                    row = elem[0]
                    col = elem[1]
                    mask[row][col] = 1
                else:
                    # import pdb; pdb.set_trace()
                    print("Error, one number cannot be found after exhausting all values")
    #if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
    #    with open(args.sp_gs_output_ptr+"_"+name+".txt",'a') as f:
    #        f.write("{} ".format(num_packs))
    return num_packs

def update_tiling_mask(sum_array, mask, threshold, num_buckets, elem_per_row, args, name):
    def get_value(v):
        return v[2]

    num_rows = sum_array.shape[0]
    num_cols = sum_array.shape[1]
    tmp_mask = np.abs(sum_array) > threshold
    num_items = np.sum(tmp_mask)

    row_nums = np.sum(tmp_mask, 1)
    num_bucket_rows = num_buckets // elem_per_row
    # stable sort to find the rows to put together
    # currently just sort the number of entries from small to large
    dtype = [('row', int), ('num', int)]
    values = [(i, row_nums[i]) for i in range(num_rows)]
    rows = np.array(values, dtype=dtype)
    sorted_rows = np.sort(rows, kind="stable", order="num")

    new_sum_array = np.zeros(sum_array.shape)
    for i in range(num_rows):
        new_sum_array[i] = sum_array[sorted_rows[i]['row']]
    new_mask = np.zeros(sum_array.shape)
    start_row = 0
    while (start_row < num_rows):
        end_row = start_row + num_bucket_rows if start_row + num_bucket_rows < num_rows else num_rows
        update_subarray_bucket_mask(new_sum_array, new_mask, threshold, num_buckets,
                                    start_row, end_row, elem_per_row, args, name)
        start_row = end_row

    for i in range(num_rows):
        mask[sorted_rows[i]['row']] = new_mask[i]



def update_bucket_mask(sum_array, mask, threshold, num_buckets,
                       elem_per_row, tile_str, args=None, name=None):
    assert len(sum_array.shape) == 2

    if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
        with open(args.sp_gs_output_ptr+"_"+name+".txt",'a') as f:
            f.write("{} {}\n".format(sum_array.shape[0],sum_array.shape[1]))
            f.write("{}\n".format(num_buckets))
            f.write("{}\n".format(elem_per_row))
        with open(args.sp_gs_output_ptr+"_"+name+"_cmf.txt",'a') as f:
            f.write("{} {}\n".format(sum_array.shape[0],sum_array.shape[1]))
            f.write("{}\n".format(num_buckets))
            f.write("{}\n".format(elem_per_row))


    if tile_str:
        tile = eval(tile_str)
        tile_row = tile[0] if tile[0] > 0 else sum_array.shape[0]
        tile_col = tile[1] if tile[1] > 0 else sum_array.shape[1]
        for i in range(0, sum_array.shape[0], tile_row):
            for j in range(0, sum_array.shape[1], tile_col):
                srow = i
                erow = i + tile_row if i + tile_row < sum_array.shape[0] else sum_array.shape[0]
                scol = j
                ecol = j + tile_col if j + tile_col < sum_array.shape[1] else sum_array.shape[1]
                new_sum_array = sum_array[srow:erow, scol:ecol]
                new_mask = mask[srow:erow, scol:ecol]
                update_tiling_mask(new_sum_array, new_mask, threshold, num_buckets,
                                   elem_per_row, args, name)
                mask[srow:erow, scol:ecol] = new_mask
    else:
        start_idx = 0
        cmf_packs = 0
        cmf_list = []
        cdf_list = []
        while start_idx < sum_array.shape[0]:
            end_idx = start_idx + num_buckets//elem_per_row if start_idx + num_buckets//elem_per_row < sum_array.shape[0] else sum_array.shape[0]


            num_packs = update_subarray_bucket_mask(sum_array, mask, threshold, num_buckets,
                                        start_idx, end_idx, elem_per_row, args, name)
            cmf_packs += num_packs
            cmf_list.append(cmf_packs)
            cdf_list.append(num_packs)

            start_idx = end_idx

        if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
            with open(args.sp_gs_output_ptr+"_"+name+".txt",'a') as f:
                for i in cdf_list:
                    f.write("{} ".format(i))
            with open(args.sp_gs_output_ptr+"_"+name+"_cmf.txt",'a') as f:
                    for i in cmf_list:
                        f.write("{} ".format(i))
