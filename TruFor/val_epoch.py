import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import logging

import torch
from torch.nn import functional as F

from config import update_config
from config import _C as config

from datasets import MixDataset
from torch.utils.data import DataLoader

from common.losses import TruForLoss
from torch.utils.tensorboard import SummaryWriter
from common.split_params import group_weight
from common.lr_schedule import WarmUpPolyLR
from common.utils import AverageMeter
import torchvision.transforms.functional as TF
from common.metrics import computeLocalizationMetrics


parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='../output', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')

args = parser.parse_args()
update_config(config, args, "trufor_paschalis.yaml")

gpu = args.gpu
loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl, format='%(message)s')

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

torch.set_flush_denormal(True)
if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    
    
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
model = confcmx(cfg=config)
model.to(device)

checkpoint = torch.load("./ckpt/detconfcmx/best_val_loss.pth", map_location=torch.device(device))
model.load_state_dict(checkpoint['state_dict'])


def freeze_dncnn(model):
    for name, param in model.named_parameters():
        if 'dncnn' in name:
            param.requires_grad = False
    print("dncnn parameters have been frozen.")

# Call this if you need to freeze dncnn
freeze_dncnn(model)

train = MixDataset(config.DATASET.TRAIN,
                   config.DATASET.IMG_SIZE,
                   train=True,
                   class_weight=config.DATASET.CLASS_WEIGHTS)

val = MixDataset(config.DATASET.VAL,
                 config.DATASET.IMG_SIZE,
                 train=False)

logging.info(train.get_info())
train_loader = DataLoader(train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=config.WORKERS,
                          pin_memory=True)

val_loader = DataLoader(val,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config.WORKERS,
                        pin_memory=True)

criterion = TruForLoss(weights=train.class_weights.to(device), ignore_index=-1)

os.makedirs('./ckpt/{}'.format(config.MODEL.NAME), exist_ok=True)
logdir = './{}/{}'.format(config.LOG_DIR, config.MODEL.NAME)
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter('./{}/{}'.format(config.LOG_DIR, config.MODEL.NAME))


params = []
cmnext_params = []
modal_extract_params = []
cmnext_params = group_weight(cmnext_params, model, torch.nn.BatchNorm2d, config.LEARNING_RATE)

params.append(dict(params=cmnext_params[0]['params'], lr=config.LEARNING_RATE))
params.append(dict(params=cmnext_params[1]['params'], weight_decay=.0,
                   lr=config.LEARNING_RATE))

optimizer = torch.optim.SGD(params,
                            lr=config.LEARNING_RATE,
                            momentum=config.SGD_MOMENTUM,
                            weight_decay=config.WD
                            )

iters_per_epoch = len(train_loader)
iters = 0
max_iters = config.EPOCHS * iters_per_epoch
min_loss = 100

lr_schedule = WarmUpPolyLR(optimizer,
                           start_lr=config.LEARNING_RATE,
                           lr_power=config.POLY_POWER,
                           total_iters=max_iters,
                           warmup_steps=iters_per_epoch * config.WARMUP_EPOCHS)

scaler = torch.cuda.amp.GradScaler()

f1 = []
f1th = []
val_loss_avg = AverageMeter()
model.eval()
epoch = 0
pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
for step, (images, _, masks, lab) in enumerate(pbar):
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        masks = masks.squeeze(1).to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            #images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #inp = [images_norm] + modals

            pred, _, _, _ = model(images)

            val_loss = criterion(pred, masks)
            
            if val_loss is None:
                print(f"Step {step}: val_loss is None")
                print(f"pred shape: {pred.shape}, masks shape: {masks.shape}")
                print(f"pred min: {pred.min()}, max: {pred.max()}, masks min: {masks.min()}, max: {masks.max()}")
                raise
            
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"Step {step}: pred contains NaN or Inf")
                raise
            if torch.isnan(masks).any() or torch.isinf(masks).any():
                print(f"Step {step}: masks contains NaN or Inf")
                raise
            if pred is None:
                print("Model returned None for pred")
                raise

        val_loss_avg.update(val_loss.detach().item())
        gt = masks.squeeze().cpu().numpy()
        map = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
        F1_best, F1_th = computeLocalizationMetrics(map, gt)
        f1.append(F1_best)
        f1th.append(F1_th)

# Calculate values
val_loss = val_loss_avg.average()
val_f1_best = np.nanmean(f1)
val_f1_fixed = np.nanmean(f1th)

# Add values to the writer
writer.add_scalar('Val Loss', val_loss, epoch)
writer.add_scalar('Val F1 best', val_f1_best, epoch)
writer.add_scalar('Val F1 fixed', val_f1_fixed, epoch)

# Print values to the console
print(f"Val Loss: {val_loss}, Val F1 best: {val_f1_best}, Val F1 fixed: {val_f1_fixed}")


