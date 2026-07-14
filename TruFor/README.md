# TruFor

This directory contains a modified version of [TruFor](https://github.com/grip-unina/TruFor) integrated with the DiffusionPrint project. The implementation replaces the standard noiseprint++ extractor with DiffusionPrint features -- a MoCo-style contrastive fingerprint learned from DnCNN activations -- for detection and localization of diffusion-based inpainting artifacts.

---

## Setup

**1. Create a conda environment:**
```bash
conda create -n trufor python=3.11
conda activate trufor
```

**2. Install PyTorch** following the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version. For example:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**3. Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

---

## Dataset: TGIF

Training uses the [TGIF dataset](https://github.com/IDLabMedia/tgif-dataset). Follow the download and preparation instructions in the TGIF repository.

**Important -- spaces in paths:** The TGIF dataset directory structure contains spaces by default (e.g. `TGIF Dataset/...`). The dataloader in `data_core.py` uses a space `" "` as the delimiter when parsing file list `.txt` files (each line: `image_path mask_path label`). This means paths with spaces will break parsing. You have two options:

- Rename the TGIF directories to remove spaces (e.g. `TGIF_Dataset/`) -- recommended
- Modify the delimiter in `data_core.py` to use a different separator (e.g. a comma or tab) and update your `.txt` file lists accordingly

---

##  Pretrained Weights

Obtain pretrained weights from the original [TruFor repository](https://github.com/grip-unina/TruFor) for the SegFormer, DnCNN, noiseprint++, and modal extractor weights. Only diffusionprint weights are available in this repo


##  Weights

The weights for TruFor checkpoint trained on AI manipulations using diffusionprint are avaiable on [Hugginface](https://huggingface.co/giakoupg/diffusionprint)


---

## Usage

**Evaluation (single image):**
```bash
python evaluate.py --image <path> --model_path <ckpt.pth> --exp trufor_diffusionprint.yaml
```

**Evaluation (CSV batch):**
```bash
python evaluate.py --csv <file.csv> --image_col image_path --mask_col mask_path \
    --model_path <ckpt.pth> --exp trufor_diffusionprint.yaml --output results.csv
```

**Training (localization):**
```bash
python train.py --exp trufor_diffusionprint.yaml
```

**Training (detection head, phase 2):**
```bash
python train_phase_2.py --exp trufor_diffusionprint.yaml
```

---

## Citation

If you use this code please cite the original TruFor paper:

```bibtex
@InProceedings{Guillaro_2023_CVPR,
    author    = {Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and Dufour, Nicholas and Verdoliva, Luisa},
    title     = {TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20606-20615}
}
```
