# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""
from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.OUTPUT_DIR = ''
_C.LOG_DIR = 'logs'
_C.GPUS = (0,)
_C.WORKERS = 4
_C.BATCH_SIZE = 18
_C.LEARNING_RATE = 0.005
_C.SGD_MOMENTUM = 0.9
_C.WD = 0.0
_C.EPOCHS = 100
_C.WARMUP_EPOCHS = 2
_C.POLY_POWER = 0.9
_C.ACCUMULATE_ITERS = 1
_C.LOSS_SDC = False
_C.LOSS_MWS = False

# Cudnn parameters
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# Model parameters
_C.MODEL = CN()
_C.MODEL.NAME = 'detconfcmx'
_C.MODEL.PRETRAINED = ''
_C.MODEL.MODS = ('RGB','NP++')
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.EXTRA.DETECTION = None
_C.MODEL.EXTRA.CONF = False
_C.MODEL.EXTRA.NP_WEIGHTS = ''

# Dataset parameters
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN = []
_C.DATASET.VAL = []
_C.DATASET.NUM_CLASSES = 2
_C.DATASET.IMG_SIZE = None
_C.DATASET.CLASS_WEIGHTS = None
_C.DATASET.EXTRA_AUG = False
# Testing parameters
_C.TEST = CN()
_C.TEST.MODEL_FILE = ''


def update_config(cfg, args, yaml_file = 'trufor.yaml'):
    cfg.defrost()
    cfg.merge_from_file(yaml_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

