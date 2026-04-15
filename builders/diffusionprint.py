import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from .DnCNN import make_net
import copy
import math


__all__ = ['DiffusionPrint']


class DiffusionPrint(nn.Module):
    """
    Contrastive learning for generative model fingerprints.

    MoCo-style architecture with DnCNN backbone and two operating modes.

    Operating modes:
        'projector':         backbone -> GAP -> MLP projector -> cosine similarity
        'flatten_projector': backbone(out=1) -> flatten -> MLP projector -> cosine

    All modes use:
        - Momentum encoder (EMA) for positive/negative encoding
        - Queue of no-positive patch embeddings tagged by category
        - Top-k hard negative mining (cross-category)
        - Asymmetric in-batch negative masking (real-real pairs excluded)

    Optional auxiliary loss:
        Classification head (cls_lambda > 0):
            Linear classifier on projected embeddings.
            Loss = L_instance + cls_lambda * L_classification

    Frequency filtering options (inductive biases for forensic signal):
        input_highpass (bool):
            Fixed Gaussian high-pass filter applied to ALL inputs (anchors,
            positives, neg queue patches) before the backbone.
            sigma controls the Gaussian blur used to construct the residual.

        anchor_bandpass (bool):
            Asymmetric bandpass filter applied to anchors only. Positives are
            left unfiltered. Forces the model to map bandpass-filtered anchors
            to the same embedding as their full-spectrum positive counterparts.
            bandpass_low_sigma / bandpass_high_sigma control the band boundaries.

        highpass_prob (float, deprecated):
            Probabilistic Laplacian high-pass on all views. Kept for backwards
            compatibility with old checkpoints.
    """

    def __init__(self, mode='projector',
                 num_levels=17, hidden_features=64, out_channels=256,
                 projection_dim=128, projector_hidden_dim=512,
                 temperature=0.07,
                 queue_size=65536, momentum=0.999, top_k=64,
                 image_size=64, augmentations='none',
                 cls_lambda=0.0, num_classes=2,
                 highpass_prob=0.0,
                 mining_mode='binary',
                 input_highpass=False, input_highpass_sigma=1.0,
                 anchor_bandpass=False,
                 bandpass_low_sigma=3.0, bandpass_high_sigma=0.5):
        super().__init__()

        assert mode in ('projector', 'flatten_projector'), f"Unknown mode: {mode}"
        assert mining_mode in ('binary', 'multiclass'), f"Unknown mining_mode: {mining_mode}"

        self.mode = mode
        self.temperature = temperature
        self.image_size = image_size
        self.queue_size = queue_size
        self.momentum = momentum
        self.top_k = top_k
        self.cls_lambda = cls_lambda
        self.num_classes = num_classes
        self.highpass_prob = highpass_prob
        self.mining_mode = mining_mode
        self.input_highpass = input_highpass
        self.input_highpass_sigma = input_highpass_sigma
        self.anchor_bandpass = anchor_bandpass
        self.bandpass_low_sigma = bandpass_low_sigma
        self.bandpass_high_sigma = bandpass_high_sigma

        if mode == 'flatten_projector':
            out_channels = 1

        self.out_channels = out_channels

        # --- Frequency filters (fixed, non-learnable) ---

        # Legacy: Laplacian high-pass (probabilistic, deprecated)
        if highpass_prob > 0:
            laplacian = torch.tensor([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]], dtype=torch.float32)
            kernel = laplacian.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
            self.register_buffer("laplacian_kernel", kernel)

        if input_highpass:
            kernel_hp = self._make_gaussian_kernel(input_highpass_sigma)
            self.register_buffer("hp_kernel", kernel_hp)

        if anchor_bandpass:
            kernel_bp_low = self._make_gaussian_kernel(bandpass_low_sigma)
            kernel_bp_high = self._make_gaussian_kernel(bandpass_high_sigma)
            self.register_buffer("bp_low_kernel", kernel_bp_low)
            self.register_buffer("bp_high_kernel", kernel_bp_high)

        # --- Online encoder (DnCNN) ---
        self.encoder_q = make_net(
            3,
            kernels=[3] * num_levels,
            features=[hidden_features] * (num_levels - 1) + [out_channels],
            bns=[False] + [True] * (num_levels - 2) + [False],
            acts=['relu'] * (num_levels - 1) + ['linear'],
            dilats=[1] * num_levels,
            bn_momentum=0.1,
            padding=1,
        )

        flat_dim = image_size * image_size  # 1 * H * W
        if mode == 'projector':
            self.projector_q = Projector(out_channels, projector_hidden_dim, projection_dim)
            embed_dim = projection_dim
        elif mode == 'flatten_projector':
            self.projector_q = Projector(flat_dim, projector_hidden_dim, projection_dim)
            embed_dim = projection_dim

        if cls_lambda > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        self.projector_k = copy.deepcopy(self.projector_q)
        for p in self.projector_k.parameters():
            p.requires_grad = False

        self.register_buffer("queue", torch.randn(embed_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_categories", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_generator_labels", torch.zeros(queue_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.augment = self._build_augmentations(augmentations)

    # ------------------------------------------------------------------
    # Filter construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_gaussian_kernel(sigma, kernel_size=None):
        """
        Build a separable 2D Gaussian kernel as a (3, 1, kH, kW) depthwise conv
        weight (one per RGB channel, groups=3).
        """
        if kernel_size is None:
            kernel_size = 2 * math.ceil(3 * sigma) + 1
        kernel_size = max(kernel_size, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        kernel_2d = torch.outer(g, g)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
        return kernel

    def _gaussian_blur(self, x, kernel):
        """Apply depthwise Gaussian blur. Pads to maintain spatial size."""
        pad = kernel.shape[-1] // 2
        return F.conv2d(x, kernel, padding=pad, groups=3)

    def _apply_highpass(self, x):
        """Fixed high-pass filter: subtract Gaussian blur."""
        return x - self._gaussian_blur(x, self.hp_kernel)

    def _apply_bandpass(self, x):
        """Bandpass filter: difference of Gaussians."""
        low = self._gaussian_blur(x, self.bp_low_kernel)
        high = self._gaussian_blur(x, self.bp_high_kernel)
        return low - high

    def _apply_legacy_highpass(self, x):
        """Apply Laplacian high-pass filter (deprecated, kept for compat)."""
        return F.conv2d(x, self.laplacian_kernel, padding=1, groups=3)

    # ------------------------------------------------------------------

    def extract(self, x):
        """Extract dense fingerprint map for inference. Returns (B, C, H, W)."""
        if self.input_highpass:
            x = self._apply_highpass(x)
        return self.encoder_q(x)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode_q(self, x):
        """Online encoder: image -> embedding vector."""
        f = self.encoder_q(x)

        if self.mode == 'projector':
            gap = F.adaptive_avg_pool2d(f, 1).flatten(1)
            z = self.projector_q(gap)
            z = F.normalize(z, dim=1)
        elif self.mode == 'flatten_projector':
            flat = f.flatten(1)
            z = self.projector_q(flat)
            z = F.normalize(z, dim=1)

        return z

    @torch.no_grad()
    def _encode_k(self, x):
        """Momentum encoder: image -> embedding vector."""
        f = self.encoder_k(x)
        if self.mode == 'projector':
            z = F.adaptive_avg_pool2d(f, 1).flatten(1)
            z = self.projector_k(z)
            return F.normalize(z, dim=1)
        elif self.mode == 'flatten_projector':
            z = f.flatten(1)
            z = self.projector_k(z)
            return F.normalize(z, dim=1)

    # ------------------------------------------------------------------
    # MoCo mechanics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1.0 - self.momentum)
        for p_q, p_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _enqueue(self, keys, categories, generator_labels=None):
        N = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + N <= self.queue_size:
            self.queue[:, ptr:ptr + N] = keys.T
            self.queue_categories[ptr:ptr + N] = categories
            if generator_labels is not None:
                self.queue_generator_labels[ptr:ptr + N] = generator_labels
        else:
            tail = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue_categories[ptr:] = categories[:tail]
            head = N - tail
            if generator_labels is not None:
                self.queue_generator_labels[ptr:] = generator_labels[:tail]
            if head > 0:
                self.queue[:, :head] = keys[tail:].T
                self.queue_categories[:head] = categories[tail:]
                if generator_labels is not None:
                    self.queue_generator_labels[:head] = generator_labels[tail:]

        self.queue_ptr[0] = (ptr + N) % self.queue_size

    def _similarity(self, a, b):
        """Compute pairwise similarity/logits (cosine-based)."""
        if b.dim() == 3:
            return torch.einsum('bd,bkd->bk', a, b)
        return torch.matmul(a, b.T)

    def _mine_hard_negatives(self, queries, anchor_cats, anchor_gen_labels=None):
        """Top-k hard negatives from queue, cross-category."""
        B, D = queries.shape
        queue = self.queue.clone().detach().T

        sim = self._similarity(queries, queue)

        if self.mining_mode == 'multiclass' and anchor_gen_labels is not None:
            q_gen_labs = self.queue_generator_labels
            valid = (q_gen_labs.unsqueeze(0) != anchor_gen_labels.unsqueeze(1)).float()
        else:
            q_cats = self.queue_categories
            target = 1 - anchor_cats.unsqueeze(1)
            valid = (q_cats.unsqueeze(0) == target).float()

        sim = sim * valid + (1 - valid) * (-1e9)
        _, topk_idx = sim.topk(self.top_k, dim=1)

        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        queue_exp = queue.unsqueeze(0).expand(B, -1, -1)
        hard_negs = torch.gather(queue_exp, 1, topk_idx_exp)

        return hard_negs

    # ------------------------------------------------------------------
    # Augmentation pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _build_augmentations(augmentations):
        if augmentations == 'none':
            return None

        transforms_list = []

        if augmentations == 'geometric':
            transforms_list.append(T.RandomHorizontalFlip())
            transforms_list.append(T.RandomVerticalFlip())
            return T.Compose(transforms_list)

        if augmentations == 'mild':
            transforms_list.append(T.RandomApply([RandomJPEG(quality_min=70, quality_max=95)], p=0.3))
            transforms_list.append(T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.2))
            transforms_list.append(T.RandomApply([GaussianNoise(std_min=0.001, std_max=0.01)], p=0.2))

        elif augmentations == 'strong':
            transforms_list.append(T.RandomApply([RandomJPEG(quality_min=50, quality_max=90)], p=0.5))
            transforms_list.append(T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3))
            transforms_list.append(T.RandomApply([GaussianNoise(std_min=0.005, std_max=0.03)], p=0.3))

        if not transforms_list:
            return None

        return T.Compose(transforms_list)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, anchors, positives, anchor_cats,
                neg_images=None, neg_cats=None,
                anchor_gen_labels=None, neg_gen_labels=None):
        """
        Args:
            anchors: (B, 3, H, W)
            positives: (B, 3, H, W)
            anchor_cats: (B,) long, 0=real, 1=generated
            neg_images: (N, 3, H, W) no-positive patches to enqueue
            neg_cats: (N,) long
            anchor_gen_labels: (B,) long, generator labels
            neg_gen_labels: (N,) long
        Returns:
            instance_loss if cls_lambda == 0.
            Otherwise tuple: (total_loss, instance_loss, cls_loss, cls_acc)
        """
        self._momentum_update()

        # --- Geometric augmentations (both views) ---
        if self.augment is not None:
            anchors = self.augment(anchors)
            positives = self.augment(positives)

        # --- Frequency filtering ---
        if self.input_highpass:
            anchors = self._apply_highpass(anchors)
            positives = self._apply_highpass(positives)
            if neg_images is not None:
                neg_images = self._apply_highpass(neg_images)

        if self.anchor_bandpass:
            anchors = self._apply_bandpass(anchors)

        # Legacy: probabilistic Laplacian high-pass (deprecated)
        if self.highpass_prob > 0 and self.training:
            if torch.rand(1).item() < self.highpass_prob:
                anchors = self._apply_legacy_highpass(anchors)
                positives = self._apply_legacy_highpass(positives)
                if neg_images is not None:
                    neg_images = self._apply_legacy_highpass(neg_images)

        # --- Encode ---
        q = self._encode_q(anchors)

        with torch.no_grad():
            k = self._encode_k(positives)

        # Enqueue no-positive patches
        if neg_images is not None and neg_cats is not None:
            with torch.no_grad():
                neg_z = self._encode_k(neg_images)
            self._enqueue(neg_z, neg_cats, neg_gen_labels)

        # --- Positive logits: (B, 1) ---
        l_pos = torch.einsum('bd,bd->b', q, k).unsqueeze(-1)
        l_pos = l_pos / self.temperature

        # --- In-batch negatives: (B, B) ---
        l_batch = self._similarity(q, k) / self.temperature

        # Asymmetric mask: exclude self and real-real pairs
        batch_mask = torch.ones_like(l_batch)
        batch_mask.fill_diagonal_(0.0)
        real_anchors = (anchor_cats == 0).unsqueeze(1)
        real_keys = (anchor_cats == 0).unsqueeze(0)
        batch_mask = batch_mask * (~(real_anchors & real_keys)).float()
        l_batch = l_batch * batch_mask + (1.0 - batch_mask) * (-1e9)

        # --- Hard negatives from queue: (B, top_k) ---
        hard_negs = self._mine_hard_negatives(q, anchor_cats, anchor_gen_labels)
        l_hard = self._similarity(q, hard_negs) / self.temperature

        # --- Combined logits: (B, 1 + B + top_k) ---
        logits = torch.cat([l_pos, l_batch, l_hard], dim=1)

        # Stabilize
        logits_stable = logits.clone()
        logits_max, _ = logits_stable.max(dim=1, keepdim=True)
        logits_stable = logits_stable - logits_max.detach()
        logits_stable = torch.clamp(logits_stable, min=-50.0, max=50.0)

        labels = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
        instance_loss = F.cross_entropy(logits_stable, labels)

        # --- Classification loss ---
        cls_loss = None
        cls_acc = None
        if self.classifier is not None and self.cls_lambda > 0:
            cls_logits = self.classifier(q)
            if self.num_classes > 2 and anchor_gen_labels is not None:
                cls_target = anchor_gen_labels
            else:
                cls_target = anchor_cats
            cls_loss = F.cross_entropy(cls_logits, cls_target)
            cls_acc = (cls_logits.argmax(dim=1) == cls_target).float().mean()

        # --- Total loss ---
        if cls_loss is not None:
            total_loss = instance_loss + self.cls_lambda * cls_loss
            return total_loss, instance_loss, cls_loss, cls_acc

        return instance_loss

class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, eps=1e-5, affine=True),
        )

    def forward(self, x):
        return self.net(x)


class RandomJPEG(nn.Module):
    """Simulate JPEG compression on tensor input."""

    def __init__(self, quality_min=70, quality_max=95):
        super().__init__()
        self.quality_min = quality_min
        self.quality_max = quality_max

    def forward(self, x):
        import io
        from PIL import Image
        from torchvision.transforms.functional import to_pil_image, to_tensor

        quality = torch.randint(self.quality_min, self.quality_max + 1, (1,)).item()
        if x.dim() == 4:
            return torch.stack([self._jpeg_single(img, quality) for img in x])
        return self._jpeg_single(x, quality)

    @staticmethod
    def _jpeg_single(img_tensor, quality):
        from torchvision.transforms.functional import to_pil_image, to_tensor
        import io
        device = img_tensor.device
        pil_img = to_pil_image(img_tensor.cpu())
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        from PIL import Image
        pil_img = Image.open(buf)
        return to_tensor(pil_img).to(device)


class GaussianNoise(nn.Module):
    """Add random Gaussian noise to tensor."""

    def __init__(self, std_min=0.001, std_max=0.01):
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max

    def forward(self, x):
        std = torch.empty(1).uniform_(self.std_min, self.std_max).item()
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0.0, 1.0)
