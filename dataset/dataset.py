import io
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import load_dataset


GENERATOR_LABEL_MAP = {
    'none': 0,
    'sd2':  1,
    'sdxl': 2,
    'flux': 3,
}


class DiffusionPrintDatasetHF(Dataset):
    """
    HuggingFace-compatible loader for the DiffusionPrint patch dataset.
    Mirrors the API of DiffusionPrintDataset so it can be used as a drop-in
    replacement in train.py.

    The dataset is loaded from the HuggingFace Hub (or a local Parquet folder)
    and split into:
      - anchor pool: rows where has_positive=True
      - hard negative pool: rows where has_positive=False

    Positive pairs are resolved via positive_master_indices (semicolon-separated
    integers pointing to master_index values in the same dataset).

    Usage:
        dataset = DiffusionPrintDatasetHF(
            repo_id="giakoupg/diffusionprint_dataset",
            transform=transforms.ToTensor(),
            exclude_generators=["flux"],
            streaming=False,   # set True for very large datasets to avoid full download
        )

    For local Parquet folder (e.g. after running convert_to_parquet.py):
        dataset = DiffusionPrintDatasetHF(
            repo_id=None,
            local_dir="./hf_parquet",
            transform=transforms.ToTensor(),
        )
    """

    def __init__(
        self,
        repo_id=None,
        local_dir=None,
        transform=None,
        exclude_generators=None,
        streaming=False,
    ):
        if repo_id is None and local_dir is None:
            raise ValueError("Provide either repo_id (HF Hub) or local_dir (local Parquet folder).")
        if repo_id is not None and local_dir is not None:
            raise ValueError("Provide only one of repo_id or local_dir, not both.")

        self.transform = transform

        # --- Load dataset ---
        if repo_id is not None:
            print(f"Loading dataset from HF Hub: {repo_id} (streaming={streaming})")
            hf_dataset = load_dataset(repo_id, split="train", streaming=streaming)
        else:
            print(f"Loading dataset from local Parquet folder: {local_dir}")
            hf_dataset = load_dataset("parquet", data_dir=local_dir, split="train", streaming=streaming)

        if streaming:
            raise NotImplementedError(
                "Streaming mode is not supported by this loader because __len__ and "
                "random positive resolution both require full index access. "
                "Set streaming=False to download the full dataset first."
            )

        # Convert to list of dicts for fast random access
        print("Converting dataset to in-memory index (this may take a moment)...")
        self.records = hf_dataset.to_list()
        print(f"  Total rows loaded: {len(self.records)}")

        # --- Build master_index -> list position lookup ---
        # needed to resolve positive_master_indices
        self.master_index_to_pos = {
            rec["master_index"]: i for i, rec in enumerate(self.records)
        }

        # --- Filter generators ---
        if exclude_generators:
            exclude_lower = {g.lower() for g in exclude_generators}
            before = len(self.records)
            self.records = [
                r for r in self.records
                if str(r.get("generator_model", "none")).lower() not in exclude_lower
            ]
            print(f"  Excluded generators {exclude_generators}: {before} -> {len(self.records)} rows")

            # Rebuild lookup after filtering
            self.master_index_to_pos = {
                rec["master_index"]: i for i, rec in enumerate(self.records)
            }

        # --- Split into anchor pool and hard negative pool ---
        self.anchor_records = [r for r in self.records if r["has_positive"]]
        self.neg_records    = [r for r in self.records if not r["has_positive"]]

        print(f"  Anchor pool (has_positive=True):  {len(self.anchor_records)}")
        print(f"  Hard negative pool:               {len(self.neg_records)}")

        # --- Pre-parse positive_master_indices for anchor pool ---
        self.anchor_positive_indices = []
        for rec in self.anchor_records:
            raw = rec.get("positive_master_indices", "")
            if raw and raw.strip():
                indices = [int(x) for x in raw.split(";") if x.strip()]
            else:
                indices = []
            self.anchor_positive_indices.append(indices)

        # --- Pre-compute integer labels ---
        self.anchor_cat_int = [0 if r["category"] == "real" else 1 for r in self.anchor_records]
        self.anchor_gen_int = [GENERATOR_LABEL_MAP.get(str(r.get("generator_model", "none")).lower(), 0)
                               for r in self.anchor_records]

        self.neg_cat_int = [0 if r["category"] == "real" else 1 for r in self.neg_records]
        self.neg_gen_int = [GENERATOR_LABEL_MAP.get(str(r.get("generator_model", "none")).lower(), 0)
                            for r in self.neg_records]

    def _decode_image(self, record):
        """Decode PNG bytes from a record into a PIL Image, apply transform."""
        png_bytes = record["image"]
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _load_by_master_index(self, master_index):
        """Load and decode an image by its master_index value."""
        pos = self.master_index_to_pos.get(master_index)
        if pos is None:
            raise KeyError(
                f"master_index {master_index} not found in loaded dataset. "
                "It may have been excluded by exclude_generators."
            )
        return self._decode_image(self.records[pos])

    def __len__(self):
        return len(self.anchor_records)

    def __getitem__(self, idx):
        """
        Returns:
            anchor_img:      (3, H, W) tensor
            positive_img:    (3, H, W) tensor
            category:        int  (0=real, 1=generated)
            generator_label: int  (0=none, 1=sd2, 2=sdxl, 3=flux)
        """
        anchor_img = self._decode_image(self.anchor_records[idx])

        pos_indices = self.anchor_positive_indices[idx]
        chosen_master_idx = pos_indices[np.random.randint(len(pos_indices))]
        positive_img = self._load_by_master_index(chosen_master_idx)

        category        = self.anchor_cat_int[idx]
        generator_label = self.anchor_gen_int[idx]

        return anchor_img, positive_img, category, generator_label

    def sample_neg_batch(self, n):
        """
        Sample n random patches from the hard negative pool.
        Mirrors DiffusionPrintDataset.sample_neg_batch.

        Returns:
            images:           (n, 3, H, W) tensor
            categories:       (n,) long tensor
            generator_labels: (n,) long tensor
        """
        indices = np.random.choice(len(self.neg_records), size=min(n, len(self.neg_records)), replace=False)

        images = []
        for i in indices:
            images.append(self._decode_image(self.neg_records[i]))

        images           = torch.stack(images, dim=0)
        categories       = torch.tensor([self.neg_cat_int[i] for i in indices], dtype=torch.long)
        generator_labels = torch.tensor([self.neg_gen_int[i] for i in indices], dtype=torch.long)

        return images, categories, generator_labels
