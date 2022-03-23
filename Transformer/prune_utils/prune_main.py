import numpy as np
import argparse
import torch

from .prune_base import prune_parse_arguments as prune_base_parse_arguments
from .admm import ADMM, prune_parse_arguments as admm_prune_parse_arguments
from .retrain import Retrain, prune_parse_arguments as retrain_parse_arguments
from .admm import admm_adjust_learning_rate
from .multi_level_admm import MultiLevelADMM
from .multi_level_retrain import MultiLevelRetrain

from .utils_pr import prune_parse_arguments as utils_prune_parse_arguments

prune_algo = None
retrain = None


def main_prune_parse_arguments(parser):
    parser.add_argument('--sp-store-weights', type=str,
                    help="store the final weights, "
                    "maybe used by the next stage pruning")
    parser.add_argument("--sp-lars", action="store_true",
                        help="enable LARS learning rate scheduler")
    parser.add_argument('--sp-lars-trust-coef', type=float, default=0.001,
                           help="LARS trust coefficient")


def prune_parse_arguments(parser):
    main_prune_parse_arguments(parser)
    prune_base_parse_arguments(parser)
    admm_prune_parse_arguments(parser)
    utils_prune_parse_arguments(parser)
    retrain_parse_arguments(parser)


def prune_init(args, model, logger=None, fixed_model=None, pre_defined_mask=None):
    global prune_algo, retrain

    if args.sp_admm_multi:
        prune_algo = MultiLevelADMM(args, model, logger)
        return
    if args.sp_retrain:
        if args.sp_prune_before_retrain:
            # For prune before retrain, we need to also set sp-admm-sparsity-type in the command line
            # We need to set sp_admm_update_epoch for ADMM, so set it to 1.
            args.sp_admm_update_epoch = 1

            prune_algo = ADMM(args, model, logger, False)
            prune_algo.prune_harden()


        prune_algo = None
        retrain = Retrain(args, model, logger, pre_defined_mask)
        retrain.fix_layer_weight_save()
        return
    if args.sp_retrain_multi:
        prune_algo = None
        retrain = MultiLevelRetrain(args, model, logger)
    if args.sp_admm:
        prune_algo = ADMM(args, model, logger)
        return

    if args.sp_subset_progressive:
        prune_algo = ADMM(args, model, logger)
        retrain = Retrain(args, model, logger)
        return


def prune_update(epoch=0, batch_idx=0):
    if prune_algo != None:
        return prune_algo.prune_update(epoch, batch_idx)
    elif retrain != None:
        return retrain.update_mask(epoch)


def prune_update_grad(opt):
    if retrain != None:
        return retrain.update_grad(opt)

def prune_fix_layer_restore():
    if prune_algo != None:
        return
    if retrain != None:
        return retrain.fix_layer_weight_restore()

def prune_generate_small_resnet_model(mode, ratio1, ratio2):
    pass

def prune_update_loss(loss):
    if prune_algo == None:
        return loss
    return prune_algo.prune_update_loss(loss)


def prune_update_combined_loss(loss):
    if prune_algo == None:
        return loss, loss, loss
    return prune_algo.prune_update_combined_loss(loss)



def prune_harden():
    if prune_algo == None:
        return None
    return prune_algo.prune_harden()


def prune_apply_masks():
    if prune_algo:
        prune_algo.apply_masks()
    if retrain:
        retrain.apply_masks()
    else:
        return
        assert(False)

def prune_apply_masks_on_grads():
    if prune_algo:
        prune_algo.apply_masks_on_grads()
    if retrain:
        retrain.apply_masks_on_grads()
    else:
        return
        assert(False)

def prune_retrain_show_masks(debug=False):
    if retrain == None:
        print("Retrain is None!")
        return
    retrain.show_masks(debug)


def prune_store_weights():
    model = None
    args = None
    logger = None
    if prune_algo :
        model = prune_algo.model
        args = prune_algo.args
        logger = prune_algo.logger
    elif retrain:
        model = retrain.model
        args = retrain.args
        logger = retrain.logger
    else:
        return
    filename = args.sp_store_weights
    if filename is None:
        return
    variables = {}
    if logger:
        p = logger.info
    else:
        p = print
    with torch.no_grad():
        p("Storing weights to {}".format(filename))
        torch.save(model.state_dict(), filename)


def prune_store_prune_params():
    if prune_algo == None:
        return
    return prune_algo.prune_store_params()


def prune_print_sparsity(model=None, logger=None, show_sparse_only=False, compressed_view=False):
    if model is None:
        if prune_algo:
            model = prune_algo.model
        elif retrain:
            model = retrain.model
        else:
            return
    if logger:
        p = logger.info
    elif prune_algo:
        p = prune_algo.logger.info
    elif retrain:
        p = retrain.logger.info
    else:
        p = print

    if show_sparse_only:
        print("The sparsity of all params (>0.01): num_nonzeros, total_num, sparsity")
        total_nz = 0
        total = 0
        for (name, W) in model.named_parameters():
            #print(name, W.shape)
            non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            if sparsity > 0.01:
                print("{}, {}, {}, {}, {}".format(name, non_zeros.shape, num_nonzeros, total_num, sparsity))
                total_nz += num_nonzeros
                total += total_num
        if total > 0:
            print("Overall sparsity for layers with sparsity >0.01: {}".format(1 - float(total_nz)/total))
        else:
            print("All layers are dense!")
        return

    if compressed_view is True:
        total_w_num = 0
        total_w_num_nz = 0
        for (name, W) in model.named_parameters():
            if "weight" in name:
                non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
                num_nonzeros = np.count_nonzero(non_zeros)
                total_w_num_nz += num_nonzeros
                total_num = non_zeros.size
                total_w_num += total_num

        sparsity = 1 - (total_w_num_nz * 1.0) / total_w_num
        print("The sparsity of all params with 'weights' in its name: num_nonzeros, total_num, sparsity")
        print("{}, {}, {}".format(total_w_num_nz, total_w_num, sparsity))
        return

    print("The sparsity of all parameters: name, num_nonzeros, total_num, shape, sparsity")
    for (name, W) in model.named_parameters():
        non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print("{}: {}, {}, {}, [{}]".format(name, str(num_nonzeros), str(total_num), non_zeros.shape, str(sparsity)))


def prune_update_learning_rate(optimizer, epoch, args):
    if prune_algo == None:
        return None
    return admm_adjust_learning_rate(optimizer, epoch, args)


# do not use, will be deprecated
def prune_retrain_apply_masks():
    apply_masks()

def prune_generate_yaml(model, sparsity, yaml_filename=None):
    if yaml_filename is None:
        yaml_filename = 'sp_{}.yaml'.format(sparsity)
    with open(yaml_filename,'w') as f:
        f.write("prune_ratios: \n")
    num_w = 0
    for name, W in model.named_parameters():
        print(name, W.shape)
        num_w += W.detach().cpu().numpy().size
        if len(W.detach().cpu().numpy().shape) > 1:
            with open(yaml_filename,'a') as f:
                if 'module.' in name:
                    f.write("{}: {}\n".format(name[7:], sparsity))
                else:
                    f.write("{}: {}\n".format(name, sparsity))
    print("Yaml file {} generated".format(yaml_filename))
    print("Total number of parameters: {}".format(num_w))
    exit()
