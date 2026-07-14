"""
Edited in September 2022
@author: fabrizio.guillaro, davide.cozzolino

Modified to support DiffusionPrint multi-channel outputs with trainable adapter.
Adapter types:
    'simple' -- Conv2d(np_out_ch -> 3, 1x1)  [original, backward compatible]
    'deep'   -- Conv2d(np_out_ch -> 16, 3x3) + BN + ReLU + Conv2d(16 -> 3, 1x1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .utils.init_func import init_weight

import logging


def preprc_imagenet_torch(x):
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
    std  = torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    return x


def create_backbone(typ, norm_layer):
    channels = [64, 128, 320, 512]
    if typ == 'mit_b2':
        logging.info('Using backbone: Segformer-B2')
        from .encoders.dual_segformer import mit_b2 as backbone_
        backbone = backbone_(norm_fuse=norm_layer)
    else:
        raise NotImplementedError('backbone not implemented')
    return backbone, channels


def build_adapter(np_out_ch, adapter_type):
    """
    Build the adapter that projects DnCNN output to 3 channels for the Segformer branch.

    Args:
        np_out_ch: number of DnCNN output channels
        adapter_type: 'simple' or 'deep'
    Returns:
        nn.Module adapter, or None if np_out_ch <= 3
    """
    if np_out_ch <= 3:
        return None

    if adapter_type == 'simple':
        adapter = nn.Conv2d(np_out_ch, 3, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(adapter.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(adapter.bias)
        logging.info(f'NP++ adapter (simple): Conv2d({np_out_ch} -> 3, 1x1)')

    elif adapter_type == 'deep':
        adapter = nn.Sequential(
            nn.Conv2d(np_out_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=True),
        )
        nn.init.kaiming_normal_(adapter[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(adapter[3].weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(adapter[3].bias)
        logging.info(f'NP++ adapter (deep): Conv2d({np_out_ch}->16, 3x3) + BN + ReLU + Conv2d(16->3, 1x1)')

    else:
        raise NotImplementedError(f'Unknown adapter type: {adapter_type}. Use "simple" or "deep".')

    return adapter


class myEncoderDecoder(nn.Module):
    def __init__(self, cfg=None, norm_layer=nn.BatchNorm2d):
        super(myEncoderDecoder, self).__init__()

        self.norm_layer = norm_layer
        self.cfg  = cfg.MODEL.EXTRA
        self.mods = cfg.MODEL.MODS

        # Number of DnCNN output channels (1 for noiseprint++, 64 for DiffusionPrint)
        if 'NP_OUT_CHANNELS' in self.cfg:
            self.np_out_ch = self.cfg.NP_OUT_CHANNELS
        else:
            self.np_out_ch = 1

        # Adapter type: 'simple' (default, backward compatible) or 'deep'
        if 'NP_ADAPTER_TYPE' in self.cfg:
            self.adapter_type = self.cfg.NP_ADAPTER_TYPE
        else:
            self.adapter_type = 'simple'

        # Import backbone and decoder
        self.backbone, self.channels = create_backbone(self.cfg.BACKBONE, norm_layer)

        if 'CONF_BACKBONE' in self.cfg:
            self.backbone_conf, self.channels_conf = create_backbone(self.cfg.CONF_BACKBONE, norm_layer)
        else:
            self.backbone_conf = None

        if self.cfg.DECODER == 'MLPDecoder':
            logging.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(
                in_channels=self.channels, num_classes=cfg.DATASET.NUM_CLASSES,
                norm_layer=norm_layer, embed_dim=self.cfg.DECODER_EMBED_DIM
            )

            if self.cfg.CONF:
                self.decode_head_conf = DecoderHead(
                    in_channels=self.channels, num_classes=1,
                    norm_layer=norm_layer, embed_dim=self.cfg.DECODER_EMBED_DIM
                )
            else:
                self.decode_head_conf = None

            self.conf_detection = None
            if self.cfg.DETECTION is not None:
                if self.cfg.DETECTION == 'none':
                    pass
                elif self.cfg.DETECTION == 'confpool':
                    self.conf_detection = 'confpool'
                    assert self.cfg.CONF
                    self.detection = nn.Sequential(
                        nn.Linear(in_features=8, out_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=128, out_features=1),
                    )
                else:
                    raise NotImplementedError('Detection mechanism not implemented')
        else:
            raise NotImplementedError('decoder not implemented')

        # Noiseprint++ / DiffusionPrint extractor
        from models.DnCNN import make_net
        num_levels = 17
        out_channel = self.np_out_ch
        self.dncnn = make_net(
            3, kernels=[3] * num_levels,
            features=[64] * (num_levels - 1) + [out_channel],
            bns=[False] + [True] * (num_levels - 2) + [False],
            acts=['relu'] * (num_levels - 1) + ['linear'],
            dilats=[1] * num_levels,
            bn_momentum=0.1, padding=1
        )

        # Adapter
        self.np_adapter = build_adapter(self.np_out_ch, self.adapter_type)

        if self.cfg.PREPRC == 'imagenet':
            self.prepro = preprc_imagenet_torch
        else:
            assert False

        self.init_weights(pretrained=cfg.MODEL.PRETRAINED)

    def init_weights(self, pretrained=None):
        if 'NP_WEIGHTS' in self.cfg and self.cfg.NP_WEIGHTS is not None and self.cfg.NP_WEIGHTS != '':
            np_weights = self.cfg.NP_WEIGHTS
            assert os.path.isfile(np_weights), f"NP weights not found: {np_weights}"
            dat = torch.load(np_weights, map_location=torch.device('cpu'))
            logging.info(f'Loading NP weights: {np_weights}')
    
            if 'model_state_dict' in dat:
                state_dict = dat['model_state_dict']
    
                # Try encoder_q.* prefix (current DiffusionPrint naming)
                encoder_q_keys = {k: v for k, v in state_dict.items() if k.startswith('encoder_q.')}
                # Fallback: legacy dncnn_q.* prefix (old naming)
                dncnn_q_keys   = {k: v for k, v in state_dict.items() if k.startswith('dncnn_q.')}
    
                if encoder_q_keys:
                    dncnn_dict = {k[len('encoder_q.'):]: v for k, v in encoder_q_keys.items()}
                    logging.info(f'  Extracted {len(dncnn_dict)} keys from DiffusionPrint checkpoint (encoder_q)')
                elif dncnn_q_keys:
                    dncnn_dict = {k[len('dncnn_q.'):]: v for k, v in dncnn_q_keys.items()}
                    logging.info(f'  Extracted {len(dncnn_dict)} keys from DiffusionPrint checkpoint (dncnn_q, legacy)')
                else:
                    logging.warning('  No encoder_q.* or dncnn_q.* keys found in checkpoint.')
                    dncnn_dict = {}
    
                if dncnn_dict:
                    missing, unexpected = self.dncnn.load_state_dict(dncnn_dict, strict=False)
                    if missing:
                        logging.warning(f'  Missing keys in dncnn: {missing}')
                    if unexpected:
                        logging.warning(f'  Unexpected keys in dncnn: {unexpected}')
    
            elif 'network' in dat:
                self.dncnn.load_state_dict(dat['network'])
            else:
                self.dncnn.load_state_dict(dat)

        # Load backbone pretrained weights
        if pretrained:
            logging.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
            if self.backbone_conf is not None:
                self.backbone_conf.init_weights(pretrained=pretrained)

        # Init heads
        logging.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, self.cfg.BN_EPS, self.cfg.BN_MOMENTUM,
                    mode='fan_in', nonlinearity='relu')

        # Freeze DnCNN if specified in FIX_MODULES
        if 'FIX_MODULES' in self.cfg and 'NP++' in self.cfg.FIX_MODULES:
            for param in self.dncnn.parameters():
                param.requires_grad = False
            logging.info('DnCNN frozen via FIX_MODULES. Adapter remains trainable.')

    def encode_decode(self, rgb, modal_x):
        if rgb is not None:
            orisize = rgb.shape
        else:
            orisize = modal_x.shape

        x = self.backbone(rgb, modal_x)
        out, feats = self.decode_head(x, return_feats=True)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)

        if self.decode_head_conf is not None:
            if self.backbone_conf is not None:
                x_conf = self.backbone_conf(rgb, modal_x)
            else:
                x_conf = x

            conf = self.decode_head_conf(x_conf)
            conf = F.interpolate(conf, size=orisize[2:], mode='bilinear', align_corners=False)
        else:
            conf = None

        if self.conf_detection is not None:
            if self.conf_detection == 'confpool':
                from .layer_utils import weighted_statistics_pooling
                f1 = weighted_statistics_pooling(conf).view(out.shape[0], -1)
                f2 = weighted_statistics_pooling(
                    out[:, 1:2, :, :] - out[:, 0:1, :, :], F.logsigmoid(conf)
                ).view(out.shape[0], -1)
                det = self.detection(torch.cat((f1, f2), -1))
            else:
                assert False
        else:
            det = None

        return out, conf, det

    def forward(self, rgb):
        if 'NP++' in self.mods:
            if 'FIX_MODULES' in self.cfg and 'NP++' in self.cfg.FIX_MODULES:
                with torch.no_grad():
                    self.dncnn.eval()
                    modal_x = self.dncnn(rgb)
            else:
                modal_x = self.dncnn(rgb)

            if self.np_adapter is not None:
                modal_x = self.np_adapter(modal_x)
            elif self.np_out_ch == 1:
                modal_x = modal_x.repeat(1, 3, 1, 1)
            else:
                assert self.np_out_ch == 3
        else:
            modal_x = None

        if self.prepro is not None:
            rgb = self.prepro(rgb)

        out, conf, det = self.encode_decode(rgb, modal_x)
        return out, conf, det, modal_x
