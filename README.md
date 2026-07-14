# DiffusionPrint: Learning Generative Fingerprints for Diffusion-Based Inpainting Localization

This is the official repository for the paper:
**DiffusionPrint: Learning Generative Fingerprints for Diffusion-Based Inpainting Localization**
Paschalis Giakoumoglou, Symeon Papadopoulos
*CVPR Workshop on AI for Media Integrity and Security (AIMS), 2026*

[[Paper]](https://openaccess.thecvf.com/content/CVPR2026W/AIMS/papers/Giakoumoglou_DiffusionPrint_Learning_Generative_Fingerprints_for_Diffusion-Based_Inpainting_Localization_CVPRW_2026_paper.pdf) [[Dataset]](https://huggingface.co/datasets/giakoupg/diffusionprint_dataset)

---

## Overview

DiffusionPrint learns per-generator fingerprints from 64x64 image patches using a MoCo-style contrastive framework built on a DnCNN backbone. The learned features replace the noiseprint++ stream in [TruFor](https://github.com/grip-unina/TruFor) for pixel-level localization of diffusion-based inpainting artifacts.

The training pipeline uses hard negative mining over a feature queue and supports an optional classification head for generator identification. At inference, the DiffusionPrint backbone is plugged into TruFor as a drop-in feature extractor; see the `TruFor/` directory for evaluation code.

---

## Requirements

```bash
conda create -n diffusionprint python=3.11
conda activate diffusionprint
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

---

## Dataset

The DiffusionPrint patch dataset is hosted on HuggingFace: [giakoupg/diffusionprint_dataset](https://huggingface.co/datasets/giakoupg/diffusionprint_dataset).

It consists of 64x64 RGB patches extracted from real and diffusion-inpainted images across three generators (SD2, SDXL, Flux). Each patch is paired with at least one positive patch (same image region, different diffusion run) for contrastive training.

**Download via HuggingFace datasets:**
```python
from datasets import load_dataset
ds = load_dataset("giakoupg/diffusionprint_dataset", split="train")
```

**Or use the provided dataset loader directly in training** (see `dataset.py`):
```python
from dataset import DiffusionPrintDatasetHF
dataset = DiffusionPrintDatasetHF(repo_id="giakoupg/diffusionprint_dataset")
```

For training with the local `.dat` memmap format (used internally at MeVer), use `DiffusionPrintDataset` from `dataset.py` with `--train_csv` and `--patches_dir`.

---

## Training

```bash
python train_diffusionprint.py \
    --train_csv /path/to/patches.csv \
    --patches_dir /path/to/patches.dat \
    --mode projector \
    --out_channels 64 \
    --projector_hidden_dim 128 \
    --projection_dim 128 \
    --temperature 0.07 \
    --batch_size 512 \
    --lr 0.001 \
    --epochs 50 \
    --top_k 512 \
    --cls_lambda 0.5 \
    --num_classes 2 \
    --augmentations geometric \
    --save_dir ./ckpt
```

Key arguments:

| Argument | Description |
|---|---|
| `--mode` | `projector` (GAP + MLP + cosine) or `flatten_projector` |
| `--out_channels` | DnCNN output channels |
| `--top_k` | Number of hard negatives mined per batch from queue |
| `--cls_lambda` | Weight of auxiliary classification loss (0 to disable) |
| `--augmentations` | `none`, `geometric`, `mild`, or `strong` |
| `--exclude_generators` | Exclude one or more generators, e.g. `--exclude_generators flux` |
| `--input_highpass` | Apply Gaussian high-pass to all inputs before backbone |
| `--noiseprint_weights` | Initialize DnCNN from noiseprint++ weights |
| `--resume` | Resume from a checkpoint |

Checkpoints and training logs (`params.json`, `losses.csv`) are saved under `--save_dir/<run_name>/`.

---

## Pretrained Weights

Pretrained DiffusionPrint checkpoints will be released shortly. Place downloaded weights under `pretrained/diffusionprint/` and pass the path via `--noiseprint_weights` or `--pretrained` when training, or configure the path in `TruFor/config.py` for evaluation.

---

## Evaluation

Evaluation is handled by the TruFor integration in the `TruFor/` subdirectory. See [`TruFor/README.md`](TruFor/README.md) for setup and usage instructions.

Quick start:
```bash
python TruFor/evaluate.py \
    --image example.png \
    --mask example_mask.png \
    --model_path TruFor/weights/trufor_diffusionprint_tgif/ckpt.pth \
    --exp TruFor/trufor_diffusionprint.yaml \
    --save_maps --maps_dir ./maps
```

---

## Repository Structure

```
diffusionprint/
├── builders/
│   ├── DnCNN.py              # DnCNN backbone
│   ├── diffusionprint.py     # MoCo contrastive model with hard negative mining
│   └── __init__.py
├── dataset/
│   └── dataset.py            # Patch dataset loaders (memmap and HuggingFace)
├── TruFor/                   # TruFor integration (localization + evaluation)
│   └── README.md
├── train_diffusionprint.py   # Training entry point
└── requirements.txt
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{giakoumoglou2026diffusionprint,
  title={DiffusionPrint: Learning Generative Fingerprints for Diffusion-Based Inpainting Localization},
  author={Giakoumoglou, Paschalis and Papadopoulos, Symeon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8654--8664},
  year={2026}
}
```

---

## Contact

For questions or inquiries, please contact giakoupg@iti.gr
