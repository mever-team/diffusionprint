import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
import os

# Generator model -> int mapping for 4-class SupCon / classification
GENERATOR_LABEL_MAP = {
    'none': 0,   # real images
    'sd2': 1,    # Stable Diffusion 2.1
    'sdxl': 2,   # Stable Diffusion XL
    'flux': 3,   # Flux Schnell
}

class DiffusionPrintDataset(Dataset):
    """
    Dataset for DiffusionPrint contrastive learning.
    Reads image patches from a single Master Memory-Mapped (.dat) file
    with fixed-shape (N, 64, 64, 3) uint8 arrays for fast O(1) random access.

    Requires the CSV to have a 'master_index' column mapping each 'patch_path' 
    to its exact row in the .dat file.
    """

    def __init__(self, csv_path, dat_path, transform=None, exclude_generators=None):
        self.dat_path = Path(dat_path)
        self.transform = transform

        # --- Load CSV ---
        df = pd.read_csv(csv_path, dtype={'image_stem': str})
        
        if 'master_index' not in df.columns:
            raise ValueError(f"The CSV {csv_path} does not have a 'master_index' column. Run the builder script first!")

        # 1. Build a fast lookup dictionary: patch_path -> master_index
        # We do this BEFORE filtering so positive_patch_paths can still be resolved
        self.path_to_index = dict(zip(df['patch_path'], df['master_index']))

        # --- Open the Master Memory-Mapped Array ---
        # Calculate total images from file size to dynamically set the shape
        bytes_per_image = 64 * 64 * 3
        total_bytes = os.path.getsize(self.dat_path)
        num_master_images = total_bytes // bytes_per_image

        print(f"Mapping master .dat file... found {num_master_images} total images.")
        self.data = np.memmap(
            self.dat_path, 
            dtype='uint8', 
            mode='r', 
            shape=(num_master_images, 64, 64, 3)
        )

        # --- CSV Splitting and Filtering ---
        if exclude_generators:
            exclude_lower = [g.lower() for g in exclude_generators]
            before = len(df)
            mask = ~df['generator_model'].str.lower().isin(exclude_lower)
            df = df[mask].reset_index(drop=True)
            removed = before - len(df)
            print(f"  Excluded generators {exclude_generators}: removed {removed} patches ({before} -> {len(df)})")

        # Validate patch paths exist in the lookup
        missing = set(df['patch_path'].tolist()) - set(self.path_to_index.keys())
        if missing:
            print(f"  WARNING: {len(missing)} patch paths in CSV not found in index mapping")

        df_pos = df[df['has_positive'] == True].reset_index(drop=True)
        df_neg = df[df['has_positive'] == False].reset_index(drop=True)

        # --- Anchor-positive pool ---
        self.anchor_paths = df_pos['patch_path'].tolist()
        self.anchor_categories = df_pos['category'].tolist()

        self.anchor_positive_paths = []
        for _, row in df_pos.iterrows():
            pos_str = row['positive_patch_paths']
            if pd.notna(pos_str) and pos_str != 'none':
                paths = [p.strip() for p in pos_str.split(';') if p.strip()]
                self.anchor_positive_paths.append(paths)
            else:
                self.anchor_positive_paths.append([])

        self.anchor_cat_int = [0 if c == 'real' else 1 for c in self.anchor_categories]
        self.anchor_gen_int = self._map_generator_labels(df_pos)

        # --- Hard negative pool (no positives) ---
        self.neg_paths = df_neg['patch_path'].tolist()
        self.neg_categories = df_neg['category'].tolist()
        self.neg_cat_int = [0 if c == 'real' else 1 for c in self.neg_categories]
        self.neg_gen_int = self._map_generator_labels(df_neg)

        print(f"Dataset loaded from {csv_path}")
        print(f"  Anchor pool (has_positive=True): {len(self.anchor_paths)}")
        print(f"  Hard negative pool (has_positive=False): {len(self.neg_paths)}")

    @staticmethod
    def _map_generator_labels(df):
        """Map generator_model column to int labels."""
        if 'generator_model' not in df.columns:
            return [0 if c == 'real' else 1 for c in df['category'].tolist()]
        labels = []
        for gm in df['generator_model'].tolist():
            gm_str = str(gm).strip().lower() if pd.notna(gm) else 'none'
            labels.append(GENERATOR_LABEL_MAP.get(gm_str, 0))
        return labels

    def _load_image(self, patch_path):
        """Load an image from Memmap by its patch_path. O(1) access."""
        master_idx = self.path_to_index[patch_path]
        arr = self.data[master_idx]  # Instantly read from disk via OS mapping
        
        image = Image.fromarray(arr, mode="RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, idx):
        """
        Returns:
            anchor_img: (3, H, W)
            positive_img: (3, H, W)
            category: int (0=real, 1=generated)
            generator_label: int (0=none, 1=sd2, 2=sdxl, 3=flux)
        """
        anchor_img = self._load_image(self.anchor_paths[idx])

        pos_paths = self.anchor_positive_paths[idx]
        pos_path = pos_paths[np.random.randint(len(pos_paths))]
        positive_img = self._load_image(pos_path)

        category = self.anchor_cat_int[idx]
        generator_label = self.anchor_gen_int[idx]

        return anchor_img, positive_img, category, generator_label

    def sample_neg_batch(self, n):
        """
        Sample n random patches from the no-positive pool (Optimized for Memmap).
        """
        # 1. Get random local indices
        indices = np.random.choice(len(self.neg_paths), size=min(n, len(self.neg_paths)), replace=False)
        
        # 2. Map local indices to master .dat file indices in one go
        master_indices = [self.path_to_index[self.neg_paths[i]] for i in indices]
        
        # 3. VECTORIZED READ: Grab all N images from disk instantly! 
        # This bypasses the Python loop and is massively faster.
        raw_images_np = self.data[master_indices]  # Shape: (N, 64, 64, 3)
        
        # 4. Apply transforms (unfortunately PIL still needs a loop, but disk I/O is done)
        images = []
        for i in range(len(raw_images_np)):
            img = Image.fromarray(raw_images_np[i], mode="RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        images = torch.stack(images, dim=0)
        
        # 5. Get labels
        categories = torch.tensor([self.neg_cat_int[i] for i in indices], dtype=torch.long)
        generator_labels = torch.tensor([self.neg_gen_int[i] for i in indices], dtype=torch.long)
        
        return images, categories, generator_labels