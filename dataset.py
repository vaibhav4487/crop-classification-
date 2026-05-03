# =============================================================================
# dataset.py
# Custom PyTorch Dataset and DataLoader for Agricultural Crop Classification.
#
# Features:
#   - Supports JPG, PNG, WEBP, BMP, TIFF (and more) via PIL
#   - Gracefully skips corrupted / unreadable images
#   - Reads resize dimensions and normalisation stats from config.py
#   - Applies augmentation during training (configurable in config.py)
#   - Provides EDA helpers: class distribution, size distribution, format counts
# =============================================================================

import os
import collections
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

import config


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def build_label_map(root_dir: str) -> dict[str, int]:
    """
    Scans *root_dir* for sub-directories (one per class) and returns a
    sorted mapping  { class_name: integer_label }.
    Sorting ensures reproducibility across runs.
    """
    classes = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    return {cls: idx for idx, cls in enumerate(classes)}


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def _get_train_transforms() -> T.Compose:
    """Augmented pipeline for the training split."""
    steps = [
        T.Resize((config.resize_y, config.resize_x)),
        T.RandomHorizontalFlip(p=config.RANDOM_FLIP_P),
        T.RandomRotation(degrees=config.RANDOM_ROTATION),
    ]
    if config.COLOR_JITTER:
        steps.append(T.ColorJitter(**config.COLOR_JITTER_PARAMS))
    steps += [
        T.ToTensor(),
        T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ]
    return T.Compose(steps)


def _get_eval_transforms() -> T.Compose:
    """Deterministic pipeline for validation / test / inference."""
    return T.Compose([
        T.Resize((config.resize_y, config.resize_x)),
        T.ToTensor(),
        T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Custom Dataset
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """
    Custom Dataset for agricultural crop classification.

    Directory layout expected inside *root_dir*:
        root_dir/
            class_name_1/
                img001.jpg
                img002.png
                ...
            class_name_2/
                ...

    Parameters
    ----------
    root_dir  : str  – path to the dataset root
    split     : str  – one of "train" | "val" | "test"
    label_map : dict – { class_name: int } produced by build_label_map()
    augment   : bool – whether to apply training augmentation
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        label_map: dict | None = None,
        augment: bool = False,
    ):
        self.root_dir  = root_dir
        self.split     = split
        self.label_map = label_map or build_label_map(root_dir)
        self.transform = _get_train_transforms() if augment else _get_eval_transforms()

        self.samples: list[tuple[str, int]] = []   # (image_path, label)
        self._load_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_samples(self):
        """Walk root_dir and collect all valid image paths with their labels."""
        for class_name, label in self.label_map.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                ext = Path(fname).suffix.lower()
                if ext in config.SUPPORTED_EXTENSIONS:
                    self.samples.append((os.path.join(class_dir, fname), label))

        # Deterministic shuffle before train/val/test split
        random.seed(config.RANDOM_SEED)
        random.shuffle(self.samples)
        self.samples = self._apply_split(self.samples)

    def _apply_split(self, samples: list) -> list:
        """Return the subset of *samples* corresponding to self.split."""
        n     = len(samples)
        n_tr  = int(n * config.TRAIN_SPLIT)
        n_val = int(n * config.VAL_SPLIT)

        if self.split == "train":
            return samples[:n_tr]
        elif self.split == "val":
            return samples[n_tr: n_tr + n_val]
        elif self.split == "test":
            return samples[n_tr + n_val:]
        else:
            raise ValueError(f"Unknown split '{self.split}'. Use 'train', 'val', or 'test'.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            # Open with PIL; convert to RGB handles grayscale, RGBA, palette modes
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError, Exception) as exc:
            # Skip corrupted image by returning a black tensor + label=-1
            # The custom collate_fn below filters these out.
            print(f"[WARNING] Skipping corrupted image: {img_path} — {exc}")
            dummy = torch.zeros(config.input_channels, config.resize_y, config.resize_x)
            return dummy, -1

        tensor = self.transform(image)
        return tensor, label

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def classes(self) -> list[str]:
        """Sorted list of class names."""
        return sorted(self.label_map, key=self.label_map.get)

    def get_label_name(self, label: int) -> str:
        inv = {v: k for k, v in self.label_map.items()}
        return inv.get(label, "unknown")


# ---------------------------------------------------------------------------
# Custom collate function (filters out corrupted-image sentinels)
# ---------------------------------------------------------------------------

def _safe_collate(batch):
    """Remove samples with label == -1 (corrupted images) before batching."""
    batch = [(img, lbl) for img, lbl in batch if lbl != -1]
    if not batch:
        return None   # Caller must handle None
    return torch.utils.data.dataloader.default_collate(batch)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def the_dataloader(
    root_dir: str,
    split: str = "train",
    label_map: dict | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    augment: bool | None = None,
) -> DataLoader:
    """
    Creates and returns a DataLoader for the given split.

    Parameters
    ----------
    root_dir    : dataset root (contains one sub-dir per class)
    split       : "train" | "val" | "test"
    label_map   : optional pre-built label map (reuse across splits!)
    batch_size  : overrides config.batch_size if provided
    num_workers : overrides config.NUM_WORKERS if provided
    augment     : overrides augmentation flag; defaults to True for training
    """
    _batch_size  = batch_size  if batch_size  is not None else config.batch_size
    _num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    _augment     = augment     if augment     is not None else (split == "train" and config.AUGMENT_TRAIN)

    dataset = CropDataset(
        root_dir  = root_dir,
        split     = split,
        label_map = label_map,
        augment   = _augment,
    )

    return DataLoader(
        dataset,
        batch_size  = _batch_size,
        shuffle     = (split == "train"),
        num_workers = _num_workers,
        collate_fn  = _safe_collate,
        pin_memory  = True,
    )


# ---------------------------------------------------------------------------
# EDA utilities
# ---------------------------------------------------------------------------

def eda_class_distribution(root_dir: str) -> dict[str, int]:
    """
    Returns { class_name: image_count } for all classes in root_dir.
    Plots a horizontal bar chart.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[EDA] matplotlib not installed — skipping plot.")
        return {}

    counts = {}
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        n = sum(
            1 for f in os.listdir(class_dir)
            if Path(f).suffix.lower() in config.SUPPORTED_EXTENSIONS
        )
        counts[class_name] = n

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.35)))
    ax.barh(list(counts.keys()), list(counts.values()), color="steelblue")
    ax.set_xlabel("Image count")
    ax.set_title("Class distribution")
    plt.tight_layout()
    plt.savefig("eda_class_distribution.png", dpi=120)
    plt.show()
    print("[EDA] class distribution plot saved → eda_class_distribution.png")
    return counts


def eda_image_size_distribution(root_dir: str, sample_limit: int = 500):
    """
    Samples up to *sample_limit* images and plots width vs height scatter.
    Also prints mean / std of widths and heights.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[EDA] matplotlib / numpy not installed — skipping plot.")
        return

    widths, heights = [], []
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if Path(f).suffix.lower() in config.SUPPORTED_EXTENSIONS:
                paths.append(os.path.join(root, f))

    random.seed(config.RANDOM_SEED)
    random.shuffle(paths)
    for p in paths[:sample_limit]:
        try:
            w, h = Image.open(p).size
            widths.append(w)
            heights.append(h)
        except Exception:
            continue

    widths  = np.array(widths)
    heights = np.array(heights)
    print(f"[EDA] Width  — mean: {widths.mean():.0f}, std: {widths.std():.0f}, "
          f"min: {widths.min()}, max: {widths.max()}")
    print(f"[EDA] Height — mean: {heights.mean():.0f}, std: {heights.std():.0f}, "
          f"min: {heights.min()}, max: {heights.max()}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(widths, heights, alpha=0.3, s=10, color="darkorange")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title(f"Image size distribution (n={len(widths)} samples)")
    plt.tight_layout()
    plt.savefig("eda_image_sizes.png", dpi=120)
    plt.show()
    print("[EDA] image size plot saved → eda_image_sizes.png")


def eda_format_distribution(root_dir: str) -> dict[str, int]:
    """Returns { extension: count } and prints a pie chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[EDA] matplotlib not installed — skipping plot.")
        return {}

    ext_counts: dict[str, int] = collections.Counter()
    for root, _, files in os.walk(root_dir):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in config.SUPPORTED_EXTENSIONS:
                ext_counts[ext] += 1

    if ext_counts:
        fig, ax = plt.subplots()
        ax.pie(ext_counts.values(), labels=ext_counts.keys(), autopct="%1.1f%%")
        ax.set_title("Image format distribution")
        plt.tight_layout()
        plt.savefig("eda_formats.png", dpi=120)
        plt.show()
        print("[EDA] format distribution plot saved → eda_formats.png")
    return dict(ext_counts)


def eda_sample_visualization(root_dir: str, n_per_class: int = 1):
    """Displays one sample image per class in a grid."""
    try:
        import matplotlib.pyplot as plt
        import math
    except ImportError:
        print("[EDA] matplotlib not installed — skipping plot.")
        return

    label_map = build_label_map(root_dir)
    classes   = list(label_map.keys())
    images, titles = [], []

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        files = [
            f for f in os.listdir(cls_dir)
            if Path(f).suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]
        if not files:
            continue
        try:
            img = Image.open(os.path.join(cls_dir, files[0])).convert("RGB")
            img = img.resize((config.resize_x, config.resize_y))
            images.append(img)
            titles.append(cls)
        except Exception:
            continue

    cols = 6
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.suptitle("Sample images per class", fontsize=14)
    plt.tight_layout()
    plt.savefig("eda_samples.png", dpi=120)
    plt.show()
    print("[EDA] sample visualisation saved → eda_samples.png")


def run_full_eda(root_dir: str):
    """Convenience wrapper — runs all four EDA analyses."""
    print("=" * 60)
    print("EDA: Agricultural Crop Dataset")
    print("=" * 60)
    eda_class_distribution(root_dir)
    eda_image_size_distribution(root_dir)
    eda_format_distribution(root_dir)
    eda_sample_visualization(root_dir)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_full_eda(config.DATA_DIR)
