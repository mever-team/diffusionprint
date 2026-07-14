import os
import argparse
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import update_config
from config import _C as config
from models.cmx.builder_np_conf import myEncoderDecoder as confcmx


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class ConditionalResize(ImageOnlyTransform):
    """Resize only if the longest side exceeds max_dim."""

    def __init__(self, max_dim=1024, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.max_dim = max_dim

    def apply(self, img, **params):
        if max(img.shape[0], img.shape[1]) > self.max_dim:
            img = A.LongestMaxSize(max_size=self.max_dim)(image=img)['image']
        return img


def build_augmentation(aug_name):
    """Return an albumentations transform for the requested augmentation, or None."""
    if aug_name is None or aug_name.lower() == 'none':
        return None
    aug_map = {
        'jpeg50':  lambda: A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
        'jpeg70':  lambda: A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
        'jpeg85':  lambda: A.ImageCompression(quality_lower=85, quality_upper=85, p=1.0),
        'blur':    lambda: A.GaussianBlur(blur_limit=(5, 5), p=1.0),
        'noise':   lambda: A.GaussNoise(var_limit=(10, 50), p=1.0),
    }
    if aug_name not in aug_map:
        raise ValueError(f"Unknown augmentation '{aug_name}'. Choices: {list(aug_map.keys())}")
    return aug_map[aug_name]()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EvalDataset(Dataset):
    """
    Loads images (and optionally masks) for inference.

    Accepts either:
      - a CSV file with at least an image-path column (and optionally a mask-path column)
      - a single image path (and optionally a single mask path)
    """

    def __init__(self, image_paths, mask_paths=None, augmentation=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths if mask_paths is not None else [None] * len(image_paths)
        self.has_masks = mask_paths is not None

        self.image_transform = A.Compose([
            ConditionalResize(max_dim=1024),
            ToTensorV2(),
        ])
        self.mask_resize = ConditionalResize(max_dim=1024)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation is not None:
            image = self.augmentation(image=image)['image']

        h, w = image.shape[:2]

        if mask_path is not None and str(mask_path).lower() not in ('', 'none', 'nan'):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Could not read mask: {mask_path}")
            mask = cv2.resize(mask, (w, h))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask_path = None

        image = self.image_transform(image=image)['image']
        image = image / 256.0

        mask = self.mask_resize(image=mask)['image']
        mask = ToTensorV2()(image=mask)['image']
        mask = (mask / 255.0 > 0.1).long()

        return image, mask, image_path, (mask_path is not None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)


def compute_metrics(pred_map, gt_mask, threshold=0.5):
    """Compute pixel-level metrics between a soft prediction map and a binary GT mask."""
    pred_binary = (pred_map > threshold).flatten().astype(np.uint8)
    gt_flat = gt_mask.flatten().astype(np.uint8)

    if pred_binary.max() == 0 and gt_flat.max() == 0:
        return {
            'F1': 1.0, 'Precision': 1.0, 'Recall': 1.0,
            'Specificity': 1.0, 'Accuracy': 1.0,
            'Balanced_Accuracy': 1.0, 'AUC': float('nan'), 'IoU': 1.0,
        }

    cm = confusion_matrix(gt_flat, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    eps = 1e-32

    recall = tp / max(tp + fn, eps)
    specificity = tn / max(tn + fp, eps)
    auc = safe_auc(gt_flat, pred_map.flatten())

    return {
        'F1':               2 * tp / max(2 * tp + fn + fp, eps),
        'Precision':        tp / max(tp + fp, eps),
        'Recall':           recall,
        'Specificity':      specificity,
        'Accuracy':         (tp + tn) / max(tp + tn + fp + fn, eps),
        'Balanced_Accuracy': (recall + specificity) / 2,
        'AUC':              auc,
        'IoU':              tp / max(tp + fp + fn, eps),
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def resize_tensor(t, scale=0.5):
    import torchvision.transforms.functional as TF
    return TF.resize(t, [int(t.shape[-2] * scale), int(t.shape[-1] * scale)])


def run_model(model, images, masks, device, max_retries=3):
    """Run model with OOM-triggered resize fallback. Returns (pred_map, det_score)."""
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                masks = masks.squeeze(1).to(device, non_blocking=True)
                pred, _, det, _ = model(images)

            det_score = torch.sigmoid(det).cpu().item()
            pred_map = torch.nn.functional.softmax(pred, dim=1)[:, 1, :, :]
            pred_map = pred_map.squeeze().cpu().numpy()
            gt_mask = masks.squeeze().cpu().numpy()
            return pred_map, gt_mask, det_score

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e) and attempt < max_retries - 1:
                print(f"OOM on attempt {attempt + 1}, resizing and retrying...")
                images = resize_tensor(images)
                masks = resize_tensor(masks)
                torch.cuda.empty_cache()
            else:
                raise
    return None, None, None


def save_map(pred_map, image_path, maps_dir):
    """Save a float32 prediction map as a uint8 grayscale PNG."""
    os.makedirs(maps_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(maps_dir, stem + '_map.png')
    map_uint8 = (pred_map * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, map_uint8)
    return out_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='DiffusionPrint / TruFor evaluation script')

    # Input: single image or CSV
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                             help='Path to a single input image')
    input_group.add_argument('--csv', type=str,
                             help='Path to a CSV file with image (and optionally mask) paths')

    # Single-image mode: optional mask
    parser.add_argument('--mask', type=str, default=None,
                        help='Path to ground-truth mask (only used with --image)')

    # CSV column names
    parser.add_argument('--image_col', type=str, default='image_path',
                        help='CSV column name for image paths (default: image_path)')
    parser.add_argument('--mask_col', type=str, default=None,
                        help='CSV column name for mask paths; omit to skip mask loading')

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--exp', type=str, default='trufor_diffusionprint.yaml',
                        help='Config YAML file')

    # Output
    parser.add_argument('--output', type=str, default='results.csv',
                        help='Path to output CSV file (default: results.csv)')

    # Augmentation
    parser.add_argument('--augmentation', type=str, default=None,
                        help='Optional augmentation applied to all images. '
                             'Choices: jpeg50, jpeg70, jpeg85, blur, noise')

    # Localization maps
    parser.add_argument('--save_maps', action='store_true',
                        help='Save prediction maps as grayscale PNGs')
    parser.add_argument('--maps_dir', type=str, default='maps',
                        help='Directory to save prediction maps (default: maps)')

    # DataLoader
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=500,
                        help='Save results CSV every N steps (default: 500)')

    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help='Extra config overrides passed to update_config')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    warnings.filterwarnings('ignore')

    # Config + model
    update_config(config, args, args.exp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Loading model from {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=device, weights_only = False)
    model = confcmx(cfg=config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    if device == 'cuda':
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = False
        cudnn.deterministic = True
        cudnn.enabled = config.CUDNN.ENABLED

    # Build input lists
    if args.image is not None:
        image_paths = [args.image]
        mask_paths = [args.mask] if args.mask else None
        has_masks = args.mask is not None
    else:
        df_in = pd.read_csv(args.csv)
        if args.image_col not in df_in.columns:
            raise ValueError(f"Column '{args.image_col}' not found in CSV. "
                             f"Available columns: {list(df_in.columns)}")
        image_paths = df_in[args.image_col].tolist()

        if args.mask_col is not None:
            if args.mask_col not in df_in.columns:
                raise ValueError(f"Column '{args.mask_col}' not found in CSV. "
                                 f"Available columns: {list(df_in.columns)}")
            mask_paths = df_in[args.mask_col].tolist()
            has_masks = True
        else:
            mask_paths = None
            has_masks = False

    augmentation = build_augmentation(args.augmentation)
    if augmentation is not None:
        print(f'Augmentation: {args.augmentation}')

    dataset = EvalDataset(image_paths, mask_paths=mask_paths, augmentation=augmentation)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Result columns
    base_cols = ['image_path', 'det_score']
    metric_cols = ['F1', 'Precision', 'Recall', 'Specificity',
                   'Accuracy', 'Balanced_Accuracy', 'AUC', 'IoU']
    result_cols = base_cols + (metric_cols if has_masks else []) + (['map_path'] if args.save_maps else [])
    results = []

    for images, masks, paths, mask_available in tqdm(dataloader):
        path = paths[0]

        pred_map, gt_mask, det_score = run_model(model, images, masks, device)
        if pred_map is None:
            print(f'Skipping {path}: failed after multiple OOM retries')
            continue

        row = {'image_path': path, 'det_score': det_score}

        if args.save_maps:
            map_path = save_map(pred_map, path, args.maps_dir)
            row['map_path'] = map_path

        if has_masks and mask_available.item():
            metrics = compute_metrics(pred_map, gt_mask)
            row.update(metrics)
        elif has_masks:
            # Mask column present in CSV but this entry has no mask
            row.update({c: float('nan') for c in metric_cols})

        results.append(row)

        if len(results) % args.save_interval == 0:
            pd.DataFrame(results, columns=result_cols).to_csv(args.output, index=False)
            print(f'Checkpoint save -> {args.output} ({len(results)} rows)')

    out_df = pd.DataFrame(results, columns=result_cols)
    out_df.to_csv(args.output, index=False)
    print(f'Done. Results saved to {args.output} ({len(out_df)} rows)')

    if has_masks:
        valid = out_df.dropna(subset=['F1'])
        print(f'\nAggregate metrics over {len(valid)} images with masks:')
        for col in metric_cols:
            if col in valid.columns:
                print(f'  {col}: {valid[col].mean():.4f}')


if __name__ == '__main__':
    main()
