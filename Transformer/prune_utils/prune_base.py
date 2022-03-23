
import torch
import logging
import sys
import time
from . import utils_pr


def prune_parse_arguments(parser):
    prune_retrain = parser.add_mutually_exclusive_group()
    parser.add_argument("--sp-backbone", action="store_true",
                        help="enable sparse backboen training")
    prune_retrain.add_argument('--sp-retrain', action='store_true',
                        help="Retrain a pruned model")
    prune_retrain.add_argument('--sp-admm', action='store_true', default=False,
                        help="for admm pruning")
    prune_retrain.add_argument('--sp-admm-multi', action='store_true', default=False,
                        help="for multi-level admm pruning")
    prune_retrain.add_argument('--sp-retrain-multi', action='store_true',
                        help="For multi-level retrain a pruned model")

    parser.add_argument('--sp-config-file', type=str,
                        help="define config file")
    parser.add_argument('--sp-subset-progressive', action='store_true',
                        help="ADMM from a sparse model")
    parser.add_argument('--sp-admm-fixed-params', action='store_true',
                        help="ADMM from a sparse model, with a fixed subset of parameters")

    parser.add_argument('--sp-no-harden', action='store_true',
                        help="Do not harden the pruned matrix")
    parser.add_argument('--nv-sparse', action='store_true',
                        help="use nv's sparse library ASP")


    parser.add_argument('--sp-load-prune-params', type=str,
                        help="Load the params used in pruning only")
    parser.add_argument('--sp-store-prune-params', type=str,
                        help="Store the params used in pruning only")
    parser.add_argument('--generate-rand-seq-gap-yaml', action='store_true',
                        help="whether to generate a set of randomly selected sequential GaP yamls")

class PruneBase(object):
    def __init__(self, args, model, logger=None):
        self.args = args
        # we assume the model does not change during execution
        self.model = model
        self.configs = None
        self.prune_ratios = None

        if logger is None:
            logging.basicConfig(format='%(levelname)s:%(message)s',
                                level=logging.INFO)
            self.logger = logging.getLogger("pruning")
        else:
            self.logger = logger

        self.logger.info("Command line:")
        self.logger.info(' '.join(sys.argv))
        self.logger.info("Args:")
        self.logger.info(args)

        self.configs, self.prune_ratios = \
            utils_pr.load_configs(model, args.sp_config_file, self.logger)





    def prune_harden(self):
        self.logger.info("Hard prune")

    def prune_update(self, epoch=0, batch_idx=0):
        self.logger.info("Update prune, epoch={}, batch={}".\
            format(epoch, batch_idx))

    def prune_update_loss(self, loss):
        pass

    def prune_update_combined_loss(self, loss):
        pass

    def apply_masks(self):
        pass
    def prune_load_params(self):
        self.logger.warning("Base pruning class does not implement " +
            "load functionality")

    def prune_store_params(self):
        self.logger.warning("Base pruning class does not implement " +
            "store functionality")

    # internal function called by submodules
    def _prune_load_params(self):
        if self.args.sp_load_prune_params is None:
            return None
        self.logger.info("Loading pruning params from {}".\
            format(self.args.sp_load_prune_params))
        v = torch.load(self.args.sp_load_prune_params)
        return v

    # internal function called by submodules
    def _prune_store_params(self, variables):
        if self.args.sp_store_prune_params is None:
            return False
        self.logger.info("Storing pruning params to {}".\
            format(self.args.sp_store_prune_params))
        torch.save(variables, self.args.sp_store_prune_params)
        return True

    def _canonical_name(self, name):
        # if the model is running in parallel, the name may start
        # with "module.", but if hte model is running in a single
        # GPU, it may not, we always filter the name to be the version
        # without "module.",
        # names in the config should not start with "module."
        if "module." in name:
            return name.replace("module.", "")
        else:
            return name
