import torch
import logging
import sys
import numpy as np
import argparse
from . import utils_pr

from .retrain import Retrain


class MultiLevelRetrain(Retrain):
    def __init__(self, args, model, logger=None):
        super(MultiLevelRetrain, self).__init__(args, model, logger)
        self.configs, self.prune_ratios = \
            utils_pr.load_configs(model, args.sp_config_file, self.logger)
        frozen_ratios = self.configs["frozen_ratios"] if "frozen_ratios" in self.configs else None
        self.frozen_weights = utils_pr.get_frozen_weights(model,
            args.sp_load_frozen_weights, self.prune_ratios, frozen_ratios)

        # import pdb; pdb.set_trace()
        # in retrain, update the frozen_weights masks so that
        # the mask doesn't contain values outside the pruned large model
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in self.frozen_weights:
                continue
            if name not in self.masks:
                continue
            model_mask = self.masks[name]
            dtype = model_mask.dtype
            mask = self.frozen_weights[name]['mask'].type(dtype)
            self.frozen_weights[name]['mask'] = (mask * model_mask).type(dtype)
        

    def apply_masks(self):
        super(MultiLevelRetrain, self).apply_masks()
        utils_pr.apply_masks(self.model, self.frozen_weights)
