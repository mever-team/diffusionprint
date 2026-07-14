"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import torch
from torch import Tensor
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.eps = 1e-9
        self.ignore_index = ignore_index

    # def forward(self, logits, target):
    #     bs, c, h, w = logits.size()
    #     if bs == 0:
    #         return torch.tensor(0.).to(logits.device)
    #
    #     pred = torch.nn.functional.softmax(logits, dim=1)[:, 1, :, :]
    #
    #     not_ignored_mask = target != self.ignore_index
    #     true = target * not_ignored_mask
    #     pred = pred * not_ignored_mask
    #     dice_losses = (2. * (pred * true).sum(dim=(-1, -2, -3))) / (
    #                 (true * true).sum(dim=(-1, -2, -3)) + (pred * pred).sum(dim=(-1, -2, -3)) + self.eps)
    #     dice_loss_batch = dice_losses.mean()
    #     return 1 - dice_loss_batch
    def forward(self, logits: Tensor, target: Tensor):
        bs, c, h, w = logits.size()
        if bs == 0:
            return torch.tensor(0.).to(logits.device)

        pred = torch.softmax(logits, dim=1)[:, 1, :, :]

        not_ignored_mask = target != self.ignore_index
        true = target * not_ignored_mask
        pred = pred * not_ignored_mask

        intersection = (pred * true).sum(dim=(-1, -2))
        union = (pred + true).sum(dim=(-1, -2))

        dice_losses = 2. * intersection / (union + self.eps)
        dice_loss_batch = dice_losses.mean()

        return 1 - dice_loss_batch


        
        
class TruForLoss(torch.nn.Module):
    def __init__(self, lambda_ce: float = 0.3, ignore_index: int = -1, weights=torch.tensor([0.5, 2.5], device='cuda:0')):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.ignore_index = ignore_index
        self.criterion_bce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index)
        self.criterion_dice = DiceLoss(ignore_index=self.ignore_index)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:

        loss_bce = self.criterion_bce(logits, target)
        loss_dice = self.criterion_dice(logits, target)
        loss = self.lambda_ce * loss_bce + (1 - self.lambda_ce) * loss_dice
        return loss
        
        
class MultiClassDiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # logits: (B, C, H, W)
        # targets: (B, H, W)
        num_classes = logits.shape[1]

        # 1. Apply softmax to get probability distribution per pixel
        probs = F.softmax(logits, dim=1)

        # 2. Create a valid mask to ignore the ignore_index (usually -1)
        valid_mask = (targets != self.ignore_index).unsqueeze(1) # (B, 1, H, W)

        # 3. Safely one-hot encode the targets
        # We temporarily replace ignore_index with 0 so one_hot doesn't crash, 
        # then we mask those pixels out entirely using the valid_mask.
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore_index] = 0
        targets_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 4. Zero out ignored pixels in both predictions and targets
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # 5. Compute Dice per class
        dims = (0, 2, 3) # Compute over Batch, Height, and Width
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(targets_one_hot, dim=dims)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # 6. Return the average loss (1 - dice) across all classes
        return 1.0 - dice_score.mean()

class TruForLossMc(torch.nn.Module):
    def __init__(self, weights: Tensor, lambda_ce: float = 0.3, ignore_index: int = -1):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.ignore_index = ignore_index
        
        # Cross Entropy natively handles multi-class (B, C, H, W) vs (B, H, W)
        self.criterion_ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index)
        
        # Our new Multi-Class Dice Loss
        self.criterion_dice = MultiClassDiceLoss(ignore_index=self.ignore_index)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        loss_ce = self.criterion_ce(logits, target)
        loss_dice = self.criterion_dice(logits, target)
        
        # Blend the two losses
        loss = self.lambda_ce * loss_ce + (1 - self.lambda_ce) * loss_dice
        return loss


class TruForLossPhase2(torch.nn.Module):
    def __init__(self, lambda_det: float = 0.5, ignore_index: int = -1, pos_weight: float = None):
        super().__init__()
        self.lambda_det = lambda_det
        self.ignore_index = ignore_index
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.criterion_detect = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.criterion_conf = torch.nn.MSELoss(reduction='none')

    def to(self, device):
        super().to(device)
        if self.pos_weight is not None:
            self.criterion_detect = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.pos_weight.to(device)
            )
        return self

    def forward(self, anomaly: Tensor, gt_mask: Tensor, conf: Tensor, detect: Tensor, label: Tensor) -> Tensor:
        anomaly = torch.softmax(anomaly, dim=1)[:, 1, :, :]
        t = gt_mask * anomaly + (1 - gt_mask) * (1 - anomaly)

        valid = gt_mask != self.ignore_index
        mse = self.criterion_conf(conf.squeeze(1), t)
        Lconf = mse[valid].mean()

        Ldet = self.criterion_detect(detect.squeeze(1), label.to(torch.float32))

        loss = Lconf + Ldet * self.lambda_det

        return loss
