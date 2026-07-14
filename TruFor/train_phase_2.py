"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from common.utils import AverageMeter
from common.losses import TruForLossPhase2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
import torch
import torchvision.transforms.functional as TF


from datasets import MixDataset
from common.metrics import computeDetectionMetrics, computeLocF1_th
from common.split_params import group_weight
from common.lr_schedule import WarmUpPolyLR
from config import update_config
from config import _C as config

def set_train_mode(model):
    for name, module in model.named_modules():
        if name.startswith('decode_head_conf') or name.startswith('detection'):
            module.train()
        else:
            module.eval()

def set_train_mode_and_check_frozen(model):
    train_modules = set()
    eval_modules = set()
    frozen_modules = set()
    unfrozen_modules = set()

    for name, module in model.named_modules():
        # Set train/eval mode
        if name.startswith('decode_head_conf') or name.startswith('detection'):
            module.train()
            train_modules.add(name.split('.')[0])
        else:
            module.eval()
            eval_modules.add(name.split('.')[0])
        
        # Check if parameters are frozen
        has_params = False
        all_frozen = True
        for param in module.parameters(recurse=False):
            has_params = True
            if param.requires_grad:
                all_frozen = False
                break
        
        if has_params:
            if all_frozen:
                frozen_modules.add(name.split('.')[0])
            else:
                unfrozen_modules.add(name.split('.')[0])

    print("Modules set to train mode:")
    for module in sorted(train_modules):
        print(f"  - {module}")

    print("\nModules set to eval mode:")
    for module in sorted(eval_modules):
        print(f"  - {module}")

    print("\nModules with frozen parameters:")
    for module in sorted(frozen_modules):
        print(f"  - {module}")

    print("\nModules with unfrozen parameters:")
    for module in sorted(unfrozen_modules):
        print(f"  - {module}")

def check_module_modes(model):
    train_modules = set()
    eval_modules = set()

    for name, module in model.named_modules():
        if module.training:
            train_modules.add(name.split('.')[0])  # Add only the top-level module name
        else:
            eval_modules.add(name.split('.')[0])  # Add only the top-level module name

    print("Modules in train mode:")
    for module in sorted(train_modules):
        print(f"  - {module}")

    print("\nModules in eval mode:")
    for module in sorted(eval_modules):
        print(f"  - {module}")

def print_checkpoint_info(checkpoint):
    print("Checkpoint information:")
    for key, value in checkpoint.items():
        if key != 'state_dict':
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")

def compute_pos_weight(txt_files):
    n_auth, n_manip = 0, 0
    for path in txt_files:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                label = int(parts[-1])
                if label == 0:
                    n_auth += 1
                elif label == 1:
                    n_manip += 1
    print(f"Train set: authentic={n_auth}, manipulated={n_manip}, pos_weight={n_auth/n_manip:.4f}")
    return n_auth / n_manip
                    
parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='../output', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
parser.add_argument('-exp', '--exp', type=str, required=True, help='yaml file')

args = parser.parse_args()
#update_config(config, args, "trufor_phase_2_tgif-bal-fr.yaml")
update_config(config, args, args.exp)

gpu = args.gpu
loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl, format='%(message)s')

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED


from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
model = confcmx(cfg=config)
ckpt_path_loc = f'./ckpt/{config.MODEL.NAME}/best_val_loss.pth'
ckpt_path = f'./ckpt/{config.MODEL.NAME+"_det"}/last_ckpt.pth'

print(f"ckpt path: {ckpt_path}")
if os.path.exists(ckpt_path):
    print(f"Loading from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    last_epoch = checkpoint['epoch']+1
    model.load_state_dict(checkpoint['state_dict'])
elif os.path.exists(ckpt_path_loc):
    print(f"Loading from loc model from {ckpt_path_loc}")
    checkpoint = torch.load(ckpt_path_loc, map_location=torch.device(device))
    print_checkpoint_info(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    last_epoch = 0
    
else:
    raise ValueError("Can't load a checkpoitn")
model.to(device)


def freeze_model_except(model):
    for name, param in model.named_parameters():
        if not (name.startswith('decode_head_conf') or name.startswith('detection')):
            param.requires_grad = False
    
    print("All parameters have been frozen except for 'decode_head_conf' and 'detection'.")

# Call this if you need to freeze dncnn
freeze_model_except(model)


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

pos_weight = compute_pos_weight(config.DATASET.TRAIN)
criterion = TruForLossPhase2(pos_weight=pos_weight).to(device)

os.makedirs('./ckpt/{}'.format(config.MODEL.NAME+'_det'), exist_ok=True)
logdir = './{}/{}'.format(config.LOG_DIR, config.MODEL.NAME)
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter('./{}/{}'.format(config.LOG_DIR, config.MODEL.NAME))

cmnext_params = []
cmnext_params = group_weight(cmnext_params, model, torch.nn.BatchNorm2d, config.LEARNING_RATE)
params = cmnext_params

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
                           lr_power=0.9,
                           total_iters=max_iters,
                           warmup_steps=iters_per_epoch * config.WARMUP_EPOCHS)

scaler = torch.cuda.amp.GradScaler()


for epoch in range(last_epoch, config.EPOCHS):
    train.shuffle()  # for balanced sampling
    #model.train()
    set_train_mode(model)
    
    avg_loss = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc='Training Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
    for step, (images, _, masks, labels) in enumerate(pbar):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.squeeze(1).to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            #images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #inp = images_norm

            pred, confidence, detection, _ = model(images)
            loss = criterion(pred, masks, confidence, detection, labels) / config.ACCUMULATE_ITERS
        scaler.scale(loss).backward()
        if ((step + 1) % config.ACCUMULATE_ITERS == 0) or (step + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss.update(loss.detach().item())

        curr_iters = epoch * iters_per_epoch + step
        lr_schedule.step(cur_iter=curr_iters)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], curr_iters)

        if step == 0:
            maps = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :]
            writer.add_images('Images-Masks-Preds',
                              torch.cat((
                                  images,
                                  torch.tile(masks.unsqueeze(1), (1, 3, 1, 1)),
                                  torch.tile(maps.unsqueeze(1), (1, 3, 1, 1))), -2)
                              , epoch)

        pbar.set_postfix({"last_loss": loss.detach().item(), "epoch_loss": avg_loss.average()})
    writer.add_scalar('Training Loss', avg_loss.average(), epoch)
    
    #if (epoch + 1) % 10 == 0 or epoch == config.EPOCHS - 1:
    if 1:
        f1_loc = []
        f1_loc_msk = []
        scores = []
        labels = []
        val_loss_avg = AverageMeter()
        model.eval()
        pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
        for step, (images, _, masks, lab) in enumerate(pbar):

            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                masks = masks.squeeze(1).to(device, non_blocking=True)
                lab = lab.to(device, non_blocking=True)
                #with torch.autocast(device_type='cuda', dtype=torch.float16):
    
                #images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #inp = [images_norm] + modals

                pred, confidence, detection, _ = model(images)
                map = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
               
                gt = masks.squeeze().cpu().numpy()
                f1 = computeLocF1_th(map, gt)
                f1_loc.append(f1)
                
                det = torch.sigmoid(detection).cpu().item()
                
                if det<0.5:
                    map = np.zeros_like(map)
                f1 = computeLocF1_th(map, gt)
                f1_loc_msk.append(f1)
                
                val_loss = criterion(pred, masks, confidence, detection, lab)
                val_loss_avg.update(val_loss.detach().item())

                scores.append(det)
                labels.append(lab.squeeze().detach().cpu().item())
    
        # Calculate values
        val_loss = val_loss_avg.average()
        auc, baCC = computeDetectionMetrics(scores, labels)
        
        writer.add_scalar('Val Loss', val_loss_avg.average(), epoch)
        writer.add_scalar('Val AUC', auc, epoch)
        writer.add_scalar('Val bACC', baCC, epoch)
        
        # Print values to the console
        print(f"Epoch {epoch} - Val Loss: {val_loss}, Val AUC: {auc}, Bal Accuracy: {baCC}, F1 loc: {np.mean(f1_loc)}, F1 loc msk: {np.mean(f1_loc_msk)}")
        
        
        result = {'epoch': epoch, 'val_loss': val_loss_avg.average(), 'val_f1_best': baCC,
                  'val_auc': auc, 'state_dict': model.state_dict()}
        torch.save(result, './ckpt/{}/last_ckpt.pth'.format(config.MODEL.NAME+"_det"))
        
        if val_loss_avg.average() < min_loss:
            min_loss = val_loss_avg.average()
            result = {'epoch': epoch, 'val_loss': val_loss_avg.average(), 'val_f1_best': baCC,
                      'val_auc': auc, 'state_dict': model.state_dict()}
            torch.save(result, './ckpt/{}/best_val_loss.pth'.format(config.MODEL.NAME+"_det"))

result = {'epoch': config.EPOCHS - 1, 'val_loss': val_loss_avg.average(), 'val_baCC': baCC,
          'val_auc': auc, 'state_dict': model.state_dict()}
torch.save(result, './ckpt/{}/final.pth'.format(config.MODEL.NAME+"_det"))