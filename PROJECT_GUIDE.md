# Agricultural Crop Image Classification — Complete Project Guide

> **Course:** Image and Video Processing with Deep Learning  
> **Problem:** 30-class multi-class image classification of agricultural crops  
> **Model:** ResNet-50 backbone (ImageNet pretrained) + custom classifier head  
> **Framework:** PyTorch 2.1+ / torchvision 0.16+  
> **Python:** 3.10+ (uses `X | Y` union type syntax)

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Directory Structure — Every File Explained](#2-directory-structure--every-file-explained)
3. [Dependency Map — How Files Import Each Other](#3-dependency-map--how-files-import-each-other)
4. [Setup and Installation](#4-setup-and-installation)
5. [config.py — Line by Line](#5-configpy--line-by-line)
6. [dataset.py — Line by Line](#6-datasetpy--line-by-line)
7. [model.py — Line by Line](#7-modelpy--line-by-line)
8. [train.py — Line by Line](#8-trainpy--line-by-line)
9. [predict.py — Line by Line](#9-predictpy--line-by-line)
10. [interface.py — Line by Line](#10-interfacepy--line-by-line)
11. [generate_sample_data.py — Line by Line](#11-generate_sample_datapy--line-by-line)
12. [The Data Pipeline — Image to Tensor](#12-the-data-pipeline--image-to-tensor)
13. [The Model Architecture — Layer by Layer](#13-the-model-architecture--layer-by-layer)
14. [The Training Loop — Step by Step](#14-the-training-loop--step-by-step)
15. [The Inference Pipeline — Step by Step](#15-the-inference-pipeline--step-by-step)
16. [EDA — All Four Analyses Explained](#16-eda--all-four-analyses-explained)
17. [Every Hyperparameter — What It Is, Why That Value](#17-every-hyperparameter--what-it-is-why-that-value)
18. [The 30 Crop Classes and Their Integer Labels](#18-the-30-crop-classes-and-their-integer-labels)
19. [Checkpoint Format — What Is Saved and Loaded](#19-checkpoint-format--what-is-saved-and-loaded)
20. [Grading Script Compatibility — Full Reference](#20-grading-script-compatibility--full-reference)
21. [Running the Project — Every Command](#21-running-the-project--every-command)
22. [Common Customisations](#22-common-customisations)
23. [Troubleshooting — Every Error](#23-troubleshooting--every-error)
24. [Key Design Decisions and Why](#24-key-design-decisions-and-why)

---

## 1. What This Project Does

This project trains a deep convolutional neural network to classify images of agricultural crops into one of **30 categories** (apple, banana, blueberry, ..., wheat). Given any image file, the model returns the most likely crop class and a confidence score.

### End-to-end flow

```
Raw image on disk (any size, any format)
      ↓
dataset.py reads, decodes, resizes to 224×224, normalises
      ↓
model.py (ResNet-50 backbone) extracts 2048 visual features
      ↓
Custom head projects 2048 → 512 → 30 class scores (logits)
      ↓
train.py minimises CrossEntropyLoss, saves best weights
      ↓
predict.py loads weights, runs softmax, returns class name + confidence
```

### Why each file exists

| File | Purpose in one sentence |
|------|------------------------|
| `config.py` | Every single number, path, and flag used anywhere in the project — one place to change things |
| `dataset.py` | Reads images from disk, handles corruption, applies transforms, builds DataLoaders, runs EDA |
| `model.py` | Defines the neural network class and a factory function to build it |
| `train.py` | Runs the forward pass, backprop, scheduling, validation, and checkpointing |
| `predict.py` | Loads trained weights, preprocesses images, returns human-readable predictions |
| `interface.py` | **Only imports** — renames internal functions to the names the grading script expects |
| `generate_sample_data.py` | Creates 300 placeholder JPEG images so the directory structure is valid before real data is added |

---

## 2. Directory Structure — Every File Explained

```
project_student_name/
│
├── checkpoints/
│   ├── README.txt
│   └── final_weights.pth          ← CREATED BY train.py, does not exist yet
│
├── data/
│   ├── apple/
│   │   ├── img01.jpg              ← placeholder (replace with real images)
│   │   ├── img02.jpg
│   │   ├── img03.jpg
│   │   ├── img04.jpg
│   │   ├── img05.jpg
│   │   ├── img06.jpg
│   │   ├── img07.jpg
│   │   ├── img08.jpg
│   │   ├── img09.jpg
│   │   └── img10.jpg
│   ├── banana/
│   │   └── img01.jpg ... img10.jpg
│   ├── blueberry/
│   ├── cherry/
│   ├── corn/
│   ├── grape/
│   ├── grapefruit/
│   ├── guava/
│   ├── kiwi/
│   ├── lemon/
│   ├── lettuce/
│   ├── mango/
│   ├── melon/
│   ├── orange/
│   ├── papaya/
│   ├── peach/
│   ├── pear/
│   ├── pepper/
│   ├── pineapple/
│   ├── pomegranate/
│   ├── potato/
│   ├── raspberry/
│   ├── rice/
│   ├── soybean/
│   ├── spinach/
│   ├── strawberry/
│   ├── sugarcane/
│   ├── tomato/
│   ├── watermelon/
│   └── wheat/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── interface.py
├── generate_sample_data.py
└── requirements.txt
```

### Rules about the `data/` directory

- Every **sub-directory** inside `data/` is treated as one class
- The **name of the sub-directory** becomes the class label string (e.g., `apple`)
- The **alphabetical sort position** of that name determines the integer label (e.g., `apple` → 0)
- Files inside each sub-directory are scanned for supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`
- Files with other extensions (e.g., `.txt`, `.DS_Store`) are silently ignored
- Images do NOT need to be pre-resized — `dataset.py` resizes everything to `224×224`
- Images may be any resolution, any aspect ratio, any colour mode (grayscale, RGBA, palette) — all are converted to RGB

### The `checkpoints/` directory

- Does not contain `final_weights.pth` until training completes at least one epoch
- `train.py` creates the directory automatically if it does not exist (`os.makedirs(..., exist_ok=True)`)
- Only one file is ever saved: `final_weights.pth` — it is **overwritten** every time a new best validation accuracy is reached
- The `README.txt` explains the checkpoint format for anyone who opens the folder

---

## 3. Dependency Map — How Files Import Each Other

```
config.py
    ← imported by: dataset.py, model.py, train.py, predict.py
    ← imports:     os  (standard library only)

dataset.py
    ← imported by: train.py, predict.py, interface.py
    ← imports:     os, collections, random, pathlib
                   torch, torchvision.transforms, PIL
                   config  ← reads all image/augmentation/split/loader settings

model.py
    ← imported by: train.py, predict.py, interface.py
    ← imports:     torch, torch.nn, torchvision.models
                   config  ← reads num_classes, dropout_rate

train.py
    ← imported by: predict.py (for load_checkpoint), interface.py
    ← imports:     os, time
                   torch, torch.nn
                   config  ← reads epochs, lr, weight_decay, WEIGHTS_PATH, DATA_DIR
                   model   ← calls build_model()
                   dataset ← calls build_label_map(), the_dataloader()

predict.py
    ← imported by: interface.py
    ← imports:     os, pathlib, typing
                   torch, torch.nn.functional, torchvision.transforms, PIL
                   config  ← reads WEIGHTS_PATH, DATA_DIR, NORM_MEAN, NORM_STD,
                              resize_x/y, input_channels, SUPPORTED_EXTENSIONS
                   model   ← calls build_model()
                   dataset ← calls build_label_map()
                   train   ← calls load_checkpoint()

interface.py
    ← imported by: grading script (external)
    ← imports:     model.CropClassifier
                   train.train_model
                   predict.classify_crops
                   dataset.CropDataset, dataset.the_dataloader
                   config.batch_size, config.epochs
```

**Critical rule:** `config.py` imports nothing from this project. It is the root of the dependency tree. Every other file depends on it, but it depends on nothing but the Python standard library (`os`). This prevents circular imports.

---

## 4. Setup and Installation

### System requirements

- Python 3.10 or higher
- pip
- A CUDA-capable GPU is strongly recommended for training but not required
- At least 4 GB RAM (8 GB+ recommended for GPU training)

### Install all dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
torch>=2.1.0
torchvision>=0.16.0
Pillow>=10.0.0
matplotlib>=3.7.0
numpy>=1.24.0
```

- `torch` — core deep learning framework: tensors, autograd, optimisers, loss functions
- `torchvision` — provides ResNet-50 architecture and `transforms` pipeline
- `Pillow` — image reading and decoding for all supported formats
- `matplotlib` — used only in EDA functions in `dataset.py`
- `numpy` — used only in `eda_image_size_distribution()` for mean/std computation

### Generate the placeholder data

```bash
python generate_sample_data.py
```

This creates 300 JPEG images (10 per class, 30 classes) in `data/`. These are coloured rectangles with text — not real crops. They exist so the project runs without errors before you add real images.

### Verify the setup works

```bash
# Should print model summary and output shape
python model.py

# Should run EDA and save 4 PNG plots
python dataset.py
```

---

## 5. `config.py` — Line by Line

`config.py` is the first file imported by everything else. It has no logic — it only defines variables. **Never hardcode a number in any other file; always import from here.**

### Import

```python
import os
```

Only the standard library `os` module is needed — to build file paths dynamically relative to the script's location.

---

### Paths section

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
```

`__file__` is the path of `config.py` itself. `os.path.abspath` converts it to an absolute path. `os.path.dirname` strips the filename, leaving just the directory. So `BASE_DIR` is always the absolute path to the project root folder, no matter where you run the script from. This makes the whole project portable — you can move it to any location and paths still work.

```python
DATA_DIR = os.path.join(BASE_DIR, "data")
```

Absolute path to the `data/` folder. Used by `dataset.py` and `predict.py` as the root to scan for class sub-directories.

```python
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
```

Absolute path to `checkpoints/`. `train.py` calls `os.makedirs(CHECKPOINT_DIR, exist_ok=True)` to create this if it doesn't exist.

```python
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "final_weights.pth")
```

Full path to the saved model file. This is where `train.py` writes the checkpoint and where `predict.py` reads it from.

---

### Image / Input settings

```python
resize_x = 224   # target width (pixels)
resize_y = 224   # target height (pixels)
```

All images are resized to exactly `resize_y × resize_x` pixels before being fed into the model. The value 224 matches the input size that ResNet-50 was originally designed for (and trained on during ImageNet pretraining). Using a different size is possible — see Section 22.

```python
input_channels = 3   # RGB
```

The model's first convolutional layer expects 3 input channels (Red, Green, Blue). All images are converted to RGB by `dataset.py` and `predict.py` via `PIL.Image.convert("RGB")`, so this is always satisfied regardless of the source format.

```python
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
```

These are the per-channel mean and standard deviation of the **entire ImageNet-1K training set** (1.28 million images). When normalisation is applied:

```
output_channel = (input_channel - mean) / std
```

For example, a red pixel value of 0.5 becomes `(0.5 - 0.485) / 0.229 ≈ 0.065`.

The reason these specific values are used: the ResNet-50 backbone was pre-trained on ImageNet using these exact normalisation parameters. Its internal learned weights expect inputs in this distribution. Using different values causes a **distribution mismatch** that degrades transfer learning.

```python
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
```

A Python `set` (not a list) for O(1) membership testing. When `dataset.py` scans a directory, it checks each filename's suffix against this set. Files whose extension is not in this set are silently skipped. All extensions are lowercase; the code does `.lower()` on filenames before checking.

---

### Model section

```python
num_classes = 30
```

The number of output neurons in the final `Linear` layer of the classifier head. If you add or remove classes, change this and retrain from scratch.

```python
dropout_rate = 0.4
```

40% dropout probability in the classifier head. During training, each neuron in the 512-unit hidden layer has a 40% chance of being zeroed on every forward pass. This prevents co-adaptation of neurons and acts as a regulariser.

---

### Training section

```python
batch_size = 32
```

Number of images per mini-batch. A larger batch gives a smoother gradient estimate but uses more GPU memory. 32 is a reliable default for a 4–8 GB GPU with 224×224 images.

```python
epochs = 30
```

Total number of full passes over the training dataset. This is set to 30 to align with `lr_scheduler_T_max = epochs`, completing exactly one cosine annealing cycle.

```python
learning_rate = 1e-4
```

Initial learning rate for AdamW. `1e-4` (= 0.0001) is a standard starting LR for full fine-tuning of ResNet. Too high (e.g., `1e-2`) would destroy the pretrained features; too low (e.g., `1e-6`) would train too slowly to converge in 30 epochs.

```python
weight_decay = 1e-4
```

L2 regularisation coefficient. Added to every parameter's gradient as `grad += weight_decay * param`. Penalises large weights, preventing overfitting. `1e-4` is conservative but effective.

```python
lr_scheduler_T_max = epochs   # = 30
```

The `T_max` parameter of `CosineAnnealingLR`. Determines how many epochs constitute one cosine cycle. Setting it equal to `epochs` means the LR decays from `learning_rate` to `lr_min` over the entire training run in one smooth curve.

```python
lr_min = 1e-6
```

The floor value of the cosine LR schedule. The learning rate never goes below this. At `1e-6`, the final epochs make very fine-grained adjustments to weights.

---

### Data split section

```python
TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.10
```

After shuffling all samples (from all classes) together, the first 75% go to training, the next 15% to validation, and the last 10% to testing. These must sum to 1.0. The split is performed once, deterministically, using `RANDOM_SEED = 42` to shuffle before splitting.

---

### Augmentation section

```python
AUGMENT_TRAIN = True
```

Master on/off switch for training augmentation. Set to `False` to disable all randomness during training (useful for debugging reproducibility).

```python
RANDOM_FLIP_P = 0.5
```

Probability of horizontally flipping an image during training. A value of 0.5 means roughly half the training images are mirrored. This doubles the effective diversity of the training set for shapes that appear equally from both sides (most crops do).

```python
RANDOM_ROTATION = 15
```

Training images are randomly rotated by an angle uniformly sampled from `[-15°, +15°]`. The pixels outside the original image boundary after rotation are filled with black (value = 0). This teaches the model that orientation variation is normal.

```python
COLOR_JITTER = True
COLOR_JITTER_PARAMS = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
```

`torchvision.transforms.ColorJitter` randomly perturbs:
- `brightness=0.2`: brightness factor sampled from `[1-0.2, 1+0.2]` = `[0.8, 1.2]`
- `contrast=0.2`: contrast factor from `[0.8, 1.2]`
- `saturation=0.2`: saturation factor from `[0.8, 1.2]`
- `hue=0.05`: hue shift from `[-0.05, 0.05]` (small, as large hue shifts make plants look unnatural)

---

### Misc section

```python
NUM_WORKERS = 4
```

How many parallel CPU processes the DataLoader spawns for loading and preprocessing images while the GPU is running the forward/backward pass. On Windows this must be `0` (no multiprocessing). On Linux with 4+ CPU cores, 4 is a safe default.

```python
RANDOM_SEED = 42
```

Used by Python's `random.seed()` in `dataset.py` and `generate_sample_data.py` wherever deterministic shuffling is required. The value 42 is conventional but arbitrary.

```python
DEVICE = "cuda"
```

This variable is informational only — `train.py` and `predict.py` both auto-detect the device using `torch.cuda.is_available()` and ignore this value. It documents the intended device.

---

## 6. `dataset.py` — Line by Line

### Imports

```python
import os
import collections
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

import config
```

- `os` — directory listing (`os.listdir`, `os.path.join`, `os.walk`)
- `collections` — `collections.Counter` for format counting in EDA
- `random` — `random.seed`, `random.shuffle` for deterministic splits
- `pathlib.Path` — `.suffix` to extract file extension cleanly
- `torch` — tensor creation (`torch.zeros`) for corrupted-image sentinel
- `Dataset, DataLoader` — base classes from PyTorch
- `torchvision.transforms as T` — image transform pipeline components
- `PIL.Image, UnidentifiedImageError` — image opening and format errors
- `config` — all configuration values

---

### `build_label_map(root_dir)`

```python
def build_label_map(root_dir: str) -> dict[str, int]:
    classes = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    return {cls: idx for idx, cls in enumerate(classes)}
```

**What it does:**  
Scans `root_dir` for entries that are directories (not files). Sorts them alphabetically. Returns a dict mapping each name to its 0-based index.

**Why sorting matters:**  
Without sorting, `os.listdir` returns entries in filesystem order, which varies by OS and filesystem. Sorting ensures that `apple → 0`, `banana → 1`, ... is always the same mapping on any machine, any run.

**Example output for this project:**
```python
{
    "apple": 0, "banana": 1, "blueberry": 2, "cherry": 3, "corn": 4,
    "grape": 5, "grapefruit": 6, "guava": 7, "kiwi": 8, "lemon": 9,
    "lettuce": 10, "mango": 11, "melon": 12, "orange": 13, "papaya": 14,
    "peach": 15, "pear": 16, "pepper": 17, "pineapple": 18, "pomegranate": 19,
    "potato": 20, "raspberry": 21, "rice": 22, "soybean": 23, "spinach": 24,
    "strawberry": 25, "sugarcane": 26, "tomato": 27, "watermelon": 28, "wheat": 29
}
```

**This function is called in three places:**
1. `run_training()` in `train.py` — builds the map once, passes it to both train and val loaders
2. `CropDataset.__init__()` — if no `label_map` is passed, it builds one itself
3. `_get_model()` in `predict.py` — builds the map to create the inverse `{int: class_name}` lookup

---

### `_get_train_transforms()`

```python
def _get_train_transforms() -> T.Compose:
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
```

**Returns a `T.Compose` object** — a callable that applies each transform in `steps` in sequence.

The transforms are built from `config` values, so changing anything in `config.py` automatically changes the transform pipeline without touching this file.

`T.Resize` comes **before** the random transforms because it's wasteful to augment a 4000×3000 image and then resize it; better to resize first.

`T.ToTensor()` converts a PIL Image with integer pixel values `[0, 255]` into a `float32` tensor with values `[0.0, 1.0]` and shape `[C, H, W]`. This must come before `T.Normalize` because `Normalize` expects a tensor, not a PIL Image.

`T.Normalize` subtracts the mean and divides by the std **per channel**:
```
tensor_out[c] = (tensor_in[c] - NORM_MEAN[c]) / NORM_STD[c]
```
Result: a float tensor centred near 0, roughly in `[-2.5, +2.5]`.

The `if config.COLOR_JITTER:` check allows disabling colour jitter via `config.py` without removing it from the codebase.

---

### `_get_eval_transforms()`

```python
def _get_eval_transforms() -> T.Compose:
    return T.Compose([
        T.Resize((config.resize_y, config.resize_x)),
        T.ToTensor(),
        T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])
```

**Deterministic — no randomness.** Resize, convert to tensor, normalise. Every image processed with this pipeline produces exactly the same output every time, which is essential for reproducible validation metrics and consistent inference.

This same pipeline is used for validation, test, and inference (in `predict.py`).

---

### `class CropDataset(Dataset)`

Inherits from `torch.utils.data.Dataset`. The two required methods are `__len__` and `__getitem__`.

#### `__init__`

```python
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
    self.samples: list[tuple[str, int]] = []
    self._load_samples()
```

**`label_map or build_label_map(root_dir)`** — if a pre-built map is passed, use it; otherwise build one. The pre-built map pattern is important: when `run_training()` creates train and val loaders, it passes the **same** label map to both. This guarantees that class integer 5 means "grape" in both the training set and the validation set.

**`self.samples`** is a list of `(absolute_image_path, integer_label)` tuples. This list is populated by `_load_samples()` and is what `__getitem__` indexes into.

#### `_load_samples()`

```python
def _load_samples(self):
    for class_name, label in self.label_map.items():
        class_dir = os.path.join(self.root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            ext = Path(fname).suffix.lower()
            if ext in config.SUPPORTED_EXTENSIONS:
                self.samples.append((os.path.join(class_dir, fname), label))

    random.seed(config.RANDOM_SEED)
    random.shuffle(self.samples)
    self.samples = self._apply_split(self.samples)
```

**Step-by-step:**
1. Iterates over every `(class_name, integer_label)` pair in the label map
2. Constructs the class directory path
3. Skips (with `continue`) if that directory doesn't exist on disk
4. Lists all files in the directory
5. For each file, extracts the suffix, lowercases it, checks against `SUPPORTED_EXTENSIONS`
6. If valid, appends `(full_path, label_int)` to `self.samples`
7. Seeds `random` with `RANDOM_SEED=42` and shuffles the full list — **same shuffle on every run**
8. Calls `_apply_split` to keep only the slice belonging to `self.split`

#### `_apply_split(samples)`

```python
def _apply_split(self, samples: list) -> list:
    n     = len(samples)
    n_tr  = int(n * config.TRAIN_SPLIT)    # 75%
    n_val = int(n * config.VAL_SPLIT)      # 15%

    if self.split == "train":
        return samples[:n_tr]
    elif self.split == "val":
        return samples[n_tr: n_tr + n_val]
    elif self.split == "test":
        return samples[n_tr + n_val:]
    else:
        raise ValueError(...)
```

The split boundaries are computed with `int()`, which truncates. With 300 images:
- `n_tr = int(300 * 0.75) = 225` → train gets indices `[0, 225)`
- `n_val = int(300 * 0.15) = 45` → val gets indices `[225, 270)`
- test gets everything else → `[270, 300)` = 30 images

The `raise ValueError` on unknown split names gives a clear error message instead of silently returning an empty list.

#### `__len__`

```python
def __len__(self) -> int:
    return len(self.samples)
```

Required by PyTorch. The DataLoader uses this to know how many items exist and how to divide them into batches.

#### `__getitem__(idx)`

```python
def __getitem__(self, idx: int):
    img_path, label = self.samples[idx]
    try:
        image = Image.open(img_path).convert("RGB")
    except (UnidentifiedImageError, OSError, Exception) as exc:
        print(f"[WARNING] Skipping corrupted image: {img_path} — {exc}")
        dummy = torch.zeros(config.input_channels, config.resize_y, config.resize_x)
        return dummy, -1

    tensor = self.transform(image)
    return tensor, label
```

**The normal path:**
1. Looks up `(img_path, label)` in `self.samples` at position `idx`
2. Opens the image with PIL and calls `.convert("RGB")`:
   - If the image is grayscale (mode `"L"`), duplicates the single channel to 3 channels
   - If the image is RGBA (mode `"RGBA"`), drops the alpha channel, keeping RGB
   - If the image is palette-mode (mode `"P"`), converts to full RGB
   - If already RGB, does nothing
3. Applies the transform pipeline (resize → augment → ToTensor → Normalize)
4. Returns `(tensor, label)` — tensor shape `[3, 224, 224]`, label is an int

**The corrupted-image path:**
- If PIL raises any exception (file truncated, format unrecognised, disk error, etc.), catches the exception
- Prints a warning to stdout with the problematic path and the error message
- Returns a black zero tensor of shape `[3, 224, 224]` and label `-1`
- This sentinel value is handled by `_safe_collate` — see below

#### `classes` property

```python
@property
def classes(self) -> list[str]:
    return sorted(self.label_map, key=self.label_map.get)
```

Returns class names sorted by their integer index. So `dataset.classes[0]` returns `"apple"`, `dataset.classes[5]` returns `"grape"`, etc.

#### `get_label_name(label)`

```python
def get_label_name(self, label: int) -> str:
    inv = {v: k for k, v in self.label_map.items()}
    return inv.get(label, "unknown")
```

Converts an integer label back to a class name string. Builds the inverse dict on every call (not cached), but this function is only used for debugging/display purposes, not in a hot path.

---

### `_safe_collate(batch)`

```python
def _safe_collate(batch):
    batch = [(img, lbl) for img, lbl in batch if lbl != -1]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
```

**This is the custom `collate_fn` passed to every DataLoader in this project.**

The DataLoader calls `collate_fn` on every list of `(tensor, label)` pairs that it has sampled for one batch. Normally `default_collate` stacks them into `[B, 3, 224, 224]` and `[B]` tensors. But `default_collate` would crash or misbehave if any label is `-1` (corrupted image sentinel).

The safe collate:
1. Filters out any `(tensor, -1)` items from the batch
2. If the entire batch was corrupted and is now empty, returns `None` — the training loop checks `if batch is None: continue`
3. Otherwise, passes the clean batch to the standard `default_collate`

This mechanism means that **no matter how many images in your dataset are corrupted**, training never crashes. It just silently skips the bad images.

---

### `the_dataloader(...)`

```python
def the_dataloader(
    root_dir: str,
    split: str = "train",
    label_map: dict | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    augment: bool | None = None,
) -> DataLoader:
```

**Factory function** that builds and returns a configured DataLoader.

All parameters have sensible defaults pulled from `config.py`, but every one can be overridden per-call. This is useful for:
- Running a quick test with `batch_size=4`
- Disabling augmentation for debugging: `augment=False`
- Reducing workers for Windows compatibility: `num_workers=0`

```python
_augment = augment if augment is not None else (split == "train" and config.AUGMENT_TRAIN)
```

Auto-logic: if `augment` is not explicitly set, enable augmentation only for the training split (and only if `AUGMENT_TRAIN=True` in config). Val and test splits never get augmentation automatically.

```python
return DataLoader(
    dataset,
    batch_size  = _batch_size,
    shuffle     = (split == "train"),
    num_workers = _num_workers,
    collate_fn  = _safe_collate,
    pin_memory  = True,
)
```

- `shuffle=True` only for train — val/test should not be shuffled (ensures consistent metric computation over identical sample orderings)
- `collate_fn=_safe_collate` — every batch passes through the corruption filter
- `pin_memory=True` — allocates batch tensors in pinned (page-locked) CPU memory for faster GPU transfer

---

### EDA functions

All four functions are in `dataset.py` but are only needed for exploration, not for training or inference. They all:
- Try to import `matplotlib` and fail gracefully if it's not installed
- Save plots as PNG files in the current working directory
- Print a confirmation message

#### `eda_class_distribution(root_dir)`

Counts the number of images per class by walking each class directory and counting files with supported extensions. Produces a **horizontal bar chart** saved as `eda_class_distribution.png`. Returns `{class_name: count}`.

#### `eda_image_size_distribution(root_dir, sample_limit=500)`

Walks all class directories, collects all image paths, shuffles them with `RANDOM_SEED`, then opens up to `sample_limit=500` images just to read their `.size` attribute (width, height) — does NOT fully decode the pixel data, so it's fast. Produces:
- Console output: mean, std, min, max of widths and heights
- A **scatter plot** (width vs height) saved as `eda_image_sizes.png`

#### `eda_format_distribution(root_dir)`

Uses `collections.Counter` to count how many files of each extension exist across all class directories. Produces a **pie chart** saved as `eda_formats.png`. Returns `{".jpg": count, ".png": count, ...}`.

#### `eda_sample_visualization(root_dir)`

Opens the first valid image in each class directory, resizes it to `224×224` (purely for display), and arranges them in a grid of **6 columns** with `math.ceil(30/6) = 5` rows. Saves to `eda_samples.png`.

#### `run_full_eda(root_dir)`

Calls all four functions in sequence. This is what runs when you do `python dataset.py`.

---

## 7. `model.py` — Line by Line

### Imports

```python
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import config
```

- `torch.nn` — provides `Module`, `Sequential`, `Linear`, `BatchNorm1d`, `ReLU`, `Dropout`, `Flatten`
- `torchvision.models` — provides `resnet50`
- `ResNet50_Weights` — an enum of available pretrained weight sets for ResNet-50

---

### `class CropClassifier(nn.Module)`

#### `__init__`

```python
weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
backbone = models.resnet50(weights=weights)
```

`ResNet50_Weights.IMAGENET1K_V2` is the second (more accurate) set of ImageNet weights provided by torchvision. It achieves ~80.9% top-1 accuracy on ImageNet, slightly better than `IMAGENET1K_V1`. These weights encode learned representations of thousands of visual concepts — edges, textures, shapes, object parts — that transfer well to plant images.

```python
self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
```

`backbone.children()` is a generator of the top-level modules inside ResNet-50. Listing them:
- `[0]` Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
- `[1]` BatchNorm2d(64)
- `[2]` ReLU(inplace=True)
- `[3]` MaxPool2d(kernel_size=3, stride=2, padding=1)
- `[4]` Layer1 (Sequential of 3 Bottleneck blocks, output channels: 256)
- `[5]` Layer2 (Sequential of 4 Bottleneck blocks, output channels: 512)
- `[6]` Layer3 (Sequential of 6 Bottleneck blocks, output channels: 1024)
- `[7]` Layer4 (Sequential of 3 Bottleneck blocks, output channels: 2048)
- `[8]` AdaptiveAvgPool2d(output_size=(1, 1))
- `[9]` Linear(2048, 1000) ← the ImageNet head, **removed by `[:-1]`**

The `[:-1]` slice removes the last element (the Linear layer). Wrapping the remaining 9 modules in `nn.Sequential` creates a single callable that runs them in order.

```python
in_features = backbone.fc.in_features  # 2048
```

This reads the input size of the original FC layer from the backbone *before* it was removed. Always `2048` for ResNet-50. Using this instead of hardcoding `2048` makes the code work if you swap the backbone to ResNet-18 (which would give `512`).

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Dropout(p=dropout_rate),
    nn.Linear(512, num_classes),
)
```

**Layer-by-layer explanation:**

- `nn.Flatten()` — takes `[B, 2048, 1, 1]` from the global average pool and returns `[B, 2048]`. The two trailing `1` dimensions are spatial dimensions of size 1 after global average pooling.
- `nn.Linear(2048, 512)` — projects from 2048 features down to 512. This compression forces the network to learn a compact representation.
- `nn.BatchNorm1d(512)` — normalises the 512-dimensional feature vector across the batch. During training, it subtracts the batch mean and divides by the batch std for each of the 512 features, then applies learnable scale and shift parameters `gamma` and `beta`. During inference (`.eval()` mode), it uses running statistics accumulated during training.
- `nn.ReLU(inplace=True)` — applies `max(0, x)` elementwise. `inplace=True` modifies the tensor directly rather than allocating a new one, saving a small amount of memory.
- `nn.Dropout(p=0.4)` — during training, randomly sets 40% of the 512 activations to zero. During inference (`.eval()` mode), does nothing (all activations pass through, but are scaled by `1-p` at training time — actually, PyTorch uses the inverted dropout convention where the scaling happens at training, not inference).
- `nn.Linear(512, 30)` — the final classification layer. Maps 512 features to 30 raw scores (logits). No activation function — raw logits are what `CrossEntropyLoss` and `F.softmax` expect.

#### `_init_weights()`

```python
def _init_weights(self):
    for module in self.classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

**Only initialises the custom classifier head** — the backbone weights come from the pretrained checkpoint and are not touched here.

**Kaiming uniform** (He initialisation): initialises weights from a uniform distribution with range `[-sqrt(6/fan_in), +sqrt(6/fan_in)]`. This is theoretically optimal for layers followed by ReLU activations and prevents vanishing/exploding gradients in newly added layers.

**Bias = 0**: Standard practice. The biases are effectively adjusted during early training.

**BatchNorm1d init**: `weight=1` and `bias=0` means the BN layer starts as the identity transformation, not distorting the signal, and learns non-trivial scale/shift over time.

#### `forward(x)`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.feature_extractor(x)   # [B, 2048, 1, 1]
    logits   = self.classifier(features)   # [B, 30]
    return logits
```

The forward pass is intentionally simple — just two sequential calls. PyTorch's `autograd` handles gradient computation automatically through these operations during `loss.backward()`.

**No softmax here.** Raw logits are returned. `CrossEntropyLoss` applies log-softmax internally (more numerically stable than separate softmax + NLLLoss). `predict.py` applies `F.softmax` manually for inference to get probabilities.

#### `get_num_params()`

```python
def get_num_params(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

`p.numel()` returns the number of elements in a parameter tensor (e.g., a `[512, 2048]` weight matrix has `512 * 2048 = 1,048,576` elements). Summing over all parameters with `requires_grad=True` gives the total trainable parameter count.

For this model: approximately **23.5 million trainable parameters** (all of ResNet-50 + the new head).

---

### `build_model(pretrained=True)`

```python
def build_model(pretrained: bool = True) -> CropClassifier:
    model = CropClassifier(
        num_classes  = config.num_classes,
        pretrained   = pretrained,
        dropout_rate = config.dropout_rate,
    )
    return model
```

A one-line factory. Other modules call `build_model()` instead of `CropClassifier(...)` directly because:
- It reads hyperparameters from `config.py` automatically
- If the constructor signature ever changes, only this function needs updating

`pretrained=False` is used in `predict.py` because when loading a trained checkpoint, you don't want to waste time downloading and applying ImageNet weights that will be immediately overwritten by `load_checkpoint()`.

---

## 8. `train.py` — Line by Line

### Imports

```python
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config
from model import build_model
from dataset import build_label_map, the_dataloader
```

- `time` — used to measure wall-clock time per epoch (`time.time()`)
- `DataLoader` — type hint for function signatures

---

### `train_model(...)` — full function

This is the function aliased as `the_trainer` in `interface.py`.

#### Parameters in detail

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `nn.Module` | Yes | The model to train. Must already be constructed (e.g., via `build_model()`). |
| `num_epochs` | `int` | Yes | Total epochs to run. |
| `train_loader` | `DataLoader` | Yes | The training DataLoader. |
| `loss_fn` | callable | Yes | A loss function, e.g. `nn.CrossEntropyLoss()`. |
| `optimizer` | torch optimiser | Yes | E.g. `torch.optim.AdamW(...)`. |
| `val_loader` | `DataLoader \| None` | No | If provided, validation metrics are computed each epoch. |
| `scheduler` | LR scheduler or None | No | If provided, `.step()` is called once per epoch. |
| `device` | `str \| None` | No | `"cuda"` or `"cpu"`. Auto-detected if `None`. |
| `save_path` | `str \| None` | No | Where to save checkpoints. Defaults to `config.WEIGHTS_PATH`. |

#### Return value

```python
{
    "train_loss": [2.43, 2.01, 1.78, ...],   # list of floats, one per epoch
    "train_acc":  [0.24, 0.35, 0.44, ...],   # list of floats, one per epoch
    "val_loss":   [2.21, 1.90, 1.65, ...],   # list of floats, one per epoch
    "val_acc":    [0.28, 0.38, 0.47, ...],   # list of floats, one per epoch
}
```

Use this dict to plot training curves after training completes.

#### Device setup

```python
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

`torch.cuda.is_available()` returns `True` if a CUDA GPU is available and CUDA drivers are installed. `.to(device)` moves all model parameters and buffers to the specified device. This is an **in-place operation** — `model` now lives on GPU/CPU.

#### Epoch loop

```python
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
```

`range(1, num_epochs + 1)` means epoch numbers go from `1` to `num_epochs` inclusive (not `0` to `num_epochs - 1`), which makes the log output `[Epoch 001/030]` human-readable.

#### Training phase — `model.train()`

```python
model.train()
```

Sets the model to training mode. This affects:
- **Dropout layers**: randomly zero activations during forward pass
- **BatchNorm layers**: use batch statistics (not running statistics) for normalisation

These behaviours should only happen during training, not during validation or inference.

```python
for batch_idx, batch in enumerate(train_loader):
    if batch is None:
        continue
```

`batch` is `None` when `_safe_collate` received an entirely-corrupted mini-batch. The `continue` skips to the next batch without crashing.

```python
images, labels = batch
images = images.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)
```

`non_blocking=True` allows the CPU→GPU transfer to proceed asynchronously, overlapping with other CPU work. Requires `pin_memory=True` in the DataLoader (which is set).

```python
logits = model(images)       # forward pass → [B, 30]
loss   = loss_fn(logits, labels)   # scalar loss
```

`model(images)` calls `CropClassifier.forward(images)`. `loss_fn` is `CrossEntropyLoss` — it computes log-softmax internally, then computes negative log-likelihood for the true class.

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

- `zero_grad(set_to_none=True)`: clears gradients from the previous batch. `set_to_none=True` sets `.grad` attributes to `None` instead of zeroing them — slightly faster, uses less memory.
- `loss.backward()`: computes `d(loss)/d(param)` for every parameter via backpropagation through the entire computation graph.
- `optimizer.step()`: uses the computed gradients to update all parameters according to the AdamW update rule.

```python
running_loss += loss.item() * images.size(0)
preds         = logits.argmax(dim=1)
correct      += (preds == labels).sum().item()
total        += images.size(0)
```

- `loss.item()` extracts the scalar loss value from a 0-dimensional tensor. Multiplying by `images.size(0)` (the batch size) gives the **sum** of losses across the batch. Dividing by `total` at the end gives the true mean loss per sample (not mean of batch means, which would be biased if the last batch is smaller).
- `logits.argmax(dim=1)` gets the index of the highest logit for each sample — the predicted class.
- `(preds == labels).sum().item()` counts how many predictions match the true labels.

#### Validation phase — `_evaluate(model, val_loader, loss_fn, device)`

```python
val_loss, val_acc = 0.0, 0.0
if val_loader is not None:
    val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device)
```

Inside `_evaluate`:

```python
model.eval()
with torch.no_grad():
    ...
```

- `model.eval()`: disables Dropout and switches BatchNorm to use running statistics
- `torch.no_grad()`: disables the entire autograd engine. No computation graph is built, gradients are not tracked. This reduces memory usage by ~50% and speeds up the forward pass during evaluation.

#### LR scheduler step

```python
if scheduler is not None:
    scheduler.step()
```

Called **once per epoch** (not once per batch). For `CosineAnnealingLR`, each call advances the cosine curve by one epoch, reducing the learning rate slightly.

#### Per-epoch log line

```
[Epoch 001/030] Train Loss: 2.4321  Acc: 0.2415 | Val Loss: 2.1100  Acc: 0.3200 | LR: 1.00e-04  (45.2s)
```

- `:03d` formats the epoch number with leading zeros (001, 002, ..., 030)
- Loss values are shown to 4 decimal places (`.4f`)
- LR is shown in scientific notation (`.2e`) — useful since it changes by orders of magnitude over training
- Elapsed time is shown for each epoch

#### Checkpoint saving

```python
if val_acc >= best_val_acc:
    best_val_acc = val_acc
    _save_checkpoint(model, optimizer, epoch, val_acc, save_path)
```

The `>=` (not `>`) means the checkpoint is saved on the very first epoch regardless (since `val_acc >= 0.0` is always true initially). After that, only better-or-equal val_acc triggers a save. "Equal" triggers a save to update the epoch number metadata, which could matter if you resume training.

---

### `_save_checkpoint(model, optimizer, epoch, val_acc, path)`

```python
torch.save(
    {
        "epoch":                  epoch,
        "val_acc":                val_acc,
        "model_state_dict":       model.state_dict(),
        "optimizer_state_dict":   optimizer.state_dict(),
    },
    path,
)
```

`torch.save` serialises the dict using Python's `pickle` to a `.pth` file. The saved dict contains:
- `epoch` — which epoch produced this best val_acc
- `val_acc` — the best validation accuracy, for reference when loading
- `model.state_dict()` — an `OrderedDict` of all parameter tensors and buffers (BatchNorm running stats, etc.). This is everything needed to reconstruct exact model weights.
- `optimizer.state_dict()` — AdamW's internal state: step count, first moment (momentum), second moment (velocity) estimates for each parameter. Needed to resume training without losing momentum.

---

### `load_checkpoint(model, path, device="cpu")`

```python
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

`map_location=device` ensures the loaded tensors are placed on the right device. When loading on CPU (for inference), this prevents CUDA being required even if the model was trained on GPU.

`model.load_state_dict(...)` copies the saved weights into `model` in-place. The function returns the full checkpoint dict (including `epoch` and `val_acc`) so callers can inspect training metadata.

---

### `run_training(data_dir=None)`

A convenience wrapper that wires all components together. Calling `python train.py` runs this.

```python
label_map    = build_label_map(data_dir)
train_loader = the_dataloader(data_dir, split="train", label_map=label_map)
val_loader   = the_dataloader(data_dir, split="val",   label_map=label_map)
```

The same `label_map` is passed to both loaders — guaranteeing consistent class-to-integer mapping.

```python
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
```

`label_smoothing=0.1` replaces hard one-hot targets like `[0, 0, 1, 0, ...]` with soft targets like `[0.00345, 0.00345, 0.9, 0.00345, ...]`. The correct class gets `1 - 0.1 = 0.9`; each wrong class gets `0.1 / (30 - 1) ≈ 0.00345`. This prevents the model from learning to push logits to ±infinity and improves generalisation.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
```

`AdamW` maintains per-parameter adaptive learning rates. For each parameter `p`:
- First moment: `m = beta1 * m + (1 - beta1) * grad` (default beta1 = 0.9)
- Second moment: `v = beta2 * v + (1 - beta2) * grad²` (default beta2 = 0.999)
- Update: `p = p - lr * m / (sqrt(v) + eps) - lr * weight_decay * p`

The last term is the weight decay, properly decoupled from the adaptive gradient (this is the "W" fix in AdamW vs. Adam).

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.lr_scheduler_T_max,
    eta_min=config.lr_min,
)
```

Sets the learning rate schedule. After each `.step()` call, the LR is:
```
LR(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
```
where `t` is the current epoch. At `t=0`: `LR = lr_max = 1e-4`. At `t=15`: `LR = 5.05e-5`. At `t=30`: `LR = lr_min = 1e-6`.

---

## 9. `predict.py` — Line by Line

### Imports

```python
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

import config
from model import build_model
from dataset import build_label_map
from train import load_checkpoint
```

- `torch.nn.functional as F` — provides `F.softmax`, which is used instead of `nn.Softmax()` because it's more convenient to call functionally
- `Union` from `typing` — for the type hint `Union[str, list[str]]` (same as `str | list[str]` but more compatible)

---

### `_INFER_TRANSFORM`

```python
_INFER_TRANSFORM = T.Compose([
    T.Resize((config.resize_y, config.resize_x)),
    T.ToTensor(),
    T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
])
```

A **module-level constant** — constructed once when the module is first imported, not on every function call. This avoids the overhead of rebuilding the transform pipeline for every prediction request.

Identical to `_get_eval_transforms()` in `dataset.py` — the same deterministic preprocessing pipeline. Both read from `config.py`, so if you change `resize_x` or `NORM_MEAN`, both are updated automatically.

---

### `_model_cache`

```python
_model_cache: dict = {}   # { weights_path: (model, label_map_inv, device) }
```

A module-level dict that acts as a simple cache. The first call to `_get_model()` with a given `weights_path` loads the model from disk and stores it here. All subsequent calls return the cached tuple immediately.

This means the model file is loaded from disk **exactly once** per Python session, even if `classify_crops()` is called thousands of times in a loop.

---

### `_get_model(weights_path, data_dir, device)`

```python
if weights_path not in _model_cache:
    model = build_model(pretrained=False)
    load_checkpoint(model, path=weights_path, device=device)
    model.to(device)
    model.eval()

    label_map     = build_label_map(data_dir)
    label_map_inv = {v: k for k, v in label_map.items()}

    _model_cache[weights_path] = (model, label_map_inv, device)
```

`build_model(pretrained=False)` is used because we're about to overwrite the weights from the checkpoint — there's no point downloading ImageNet weights first.

`model.eval()` is called here and remains in effect for all future inference calls. This disables Dropout and switches BatchNorm to use running statistics (collected during training).

`label_map_inv` is `{0: "apple", 1: "banana", 2: "blueberry", ...}` — used to convert the integer predicted class back to a human-readable name.

---

### `_load_image(img_path)`

```python
def _load_image(img_path: str) -> torch.Tensor | None:
    ext = Path(img_path).suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        print(f"[predict] Unsupported format skipped: {img_path}")
        return None
    try:
        image  = Image.open(img_path).convert("RGB")
        tensor = _INFER_TRANSFORM(image)
        return tensor.unsqueeze(0)
    except (UnidentifiedImageError, OSError, Exception) as exc:
        print(f"[predict] Could not read '{img_path}': {exc}")
        return None
```

`tensor.unsqueeze(0)` adds a batch dimension: `[3, 224, 224]` → `[1, 3, 224, 224]`. This allows individual images to be stacked with `torch.cat` later in `classify_crops`.

Returns `None` for both unsupported-format and corrupted-file cases, with a descriptive print message.

---

### `classify_crops(img_paths, ...)` — the main predictor

This is the function aliased as `the_predictor` in `interface.py`.

#### Input normalisation

```python
if isinstance(img_paths, (str, Path)):
    img_paths = [str(img_paths)]
else:
    img_paths = [str(p) for p in img_paths]
```

Accepts a single string, a `pathlib.Path`, or a list of either. Converts everything to a list of strings for uniform processing.

#### Image loading phase

```python
tensors_ok  = []
paths_ok    = []
paths_error = []

for path in img_paths:
    t = _load_image(path)
    if t is not None:
        tensors_ok.append(t)
        paths_ok.append(path)
    else:
        paths_error.append(path)
```

Separates images into loadable and unloadable. Unloadable images get an `"ERROR"` result immediately. Loadable images are batched for a single GPU forward pass.

#### Batch inference

```python
batch = torch.cat(tensors_ok, dim=0).to(_device)   # [B, 3, 224, 224]

with torch.no_grad():
    logits = model(batch)                   # [B, 30]
    probs  = F.softmax(logits, dim=1)       # [B, 30]  — sums to 1 per row
    confs, preds = probs.max(dim=1)         # [B], [B]
```

`F.softmax(logits, dim=1)` converts raw logits to a probability distribution. `dim=1` means softmax is applied across the 30 class scores for each sample (not across the batch).

`probs.max(dim=1)` returns two tensors:
- `confs`: the maximum probability for each sample (the confidence score)
- `preds`: the index (class) that achieved that maximum probability

#### Result assembly

```python
for path, pred, conf, prob_vec in zip(paths_ok, preds.cpu().tolist(), confs.cpu().tolist(), probs.cpu().tolist()):
    entry = {
        "path":       path,
        "label":      label_map_inv.get(pred, f"class_{pred}"),
        "label_idx":  pred,
        "confidence": round(conf, 4),
    }
    if return_probs:
        entry["probs"] = prob_vec
    results.append(entry)
```

`.cpu().tolist()` moves tensors back to CPU (in case they're on GPU) and converts to Python lists/scalars.

`label_map_inv.get(pred, f"class_{pred}")` looks up the class name. If somehow the predicted index is not in the map (shouldn't happen), it falls back to `"class_5"` etc. rather than raising a `KeyError`.

#### Order preservation

```python
order = {p: i for i, p in enumerate(img_paths)}
results.sort(key=lambda r: order.get(r["path"], 0))
```

Because error results are appended before success results in `results`, the output list may not be in the same order as the input list. This sort restores the original input order, so `results[0]` always corresponds to `img_paths[0]`.

---

### `classify_crops_topk(img_paths, k=5, ...)`

Calls `classify_crops(..., return_probs=True)` to get the full probability vector for each image, then extracts the top-k indices by sorting `probs` in descending order.

```python
top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
```

`sorted(range(30), key=lambda i: probs[i], reverse=True)` sorts the indices 0–29 by their probability, highest first. `[:k]` takes the top k.

Returns per image: `{"path": ..., "top_labels": ["apple", "pear", ...], "top_probs": [0.92, 0.04, ...]}`.

---

### CLI entry point

```python
if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python predict.py path/to/img1.jpg path/to/img2.png ...")
    else:
        predictions = classify_crops(paths)
        for p in predictions:
            print(f"{p['path']}  →  {p['label']}  (conf: {p['confidence']:.4f})")
```

`sys.argv[1:]` collects all command-line arguments after the script name. Allows running predictions directly:
```bash
python predict.py data/apple/img01.jpg data/banana/img03.jpg
```

---

## 10. `interface.py` — Line by Line

```python
from model   import CropClassifier  as TheModel
from train   import train_model      as the_trainer
from predict import classify_crops   as the_predictor
from dataset import CropDataset      as TheDataset
from dataset import the_dataloader   as the_dataloader
from config  import batch_size       as the_batch_size
from config  import epochs           as total_epochs
```

**This file contains exactly 7 import statements and nothing else.** No functions, no classes, no logic.

The aliasing (`as TheModel`, `as the_trainer`, etc.) makes the internal names available under the standardised names the grading script expects. From outside:
```python
from interface import TheModel
model = TheModel()          # actually calls CropClassifier()
```

**Why a separate file for just imports?**  
The grading script is written once and uses standardised names. Different students use different internal names for their functions. `interface.py` is the translation layer — each student maps their internal names to the standard names without touching the original code.

---

## 11. `generate_sample_data.py` — Line by Line

### Imports

```python
import os
import random
from PIL import Image, ImageDraw, ImageFont
import config
```

`ImageDraw` — used to draw text on the PIL image.  
`ImageFont` — used to load a font for the text.

### `CROP_CLASSES`

```python
CROP_CLASSES = [
    "apple", "banana", "blueberry", "cherry", "corn",
    "grape", "grapefruit", "guava", "kiwi", "lemon",
    "lettuce", "mango", "melon", "orange", "papaya",
    "peach", "pear", "pepper", "pineapple", "pomegranate",
    "potato", "raspberry", "rice", "soybean", "spinach",
    "strawberry", "sugarcane", "tomato", "watermelon", "wheat",
]
```

The 30 class names. These must match the sub-directory names you actually use in `data/`. If your real dataset uses different names (e.g., `"Bell Pepper"` instead of `"pepper"`), update this list and rename the directories.

### `make_placeholder(class_name, index)`

```python
colour = tuple(random.randint(60, 220) for _ in range(3))
img    = Image.new("RGB", IMG_SIZE, color=colour)
```

Creates a 256×256 solid-colour image. Random RGB colour between (60,60,60) and (220,220,220) — avoiding very dark or very bright extremes.

```python
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except OSError:
    font = ImageFont.load_default()
```

Tries to load a system font at size 20. Falls back to PIL's built-in tiny bitmap font if the system font path doesn't exist (e.g., on macOS or Windows).

```python
bbox   = draw.textbbox((0, 0), text, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
x, y   = (IMG_SIZE[0] - tw) // 2, (IMG_SIZE[1] - th) // 2
draw.text((x, y), text, fill="white", font=font)
```

Computes the bounding box of the text, then centres it on the image by offsetting by `(width - text_width) // 2` in both dimensions.

### `generate()`

```python
if not os.path.exists(out_path):
    img = make_placeholder(cls, i)
    img.save(out_path, "JPEG", quality=90)
```

The `if not os.path.exists(out_path)` check means running `generate_sample_data.py` twice does not overwrite existing files. Once you replace placeholder images with real images, this script won't clobber them.

---

## 12. The Data Pipeline — Image to Tensor

### Complete flow for one training image

```
1. disk: data/apple/img_001.jpg (e.g., 640×480, JPEG, RGB)
      ↓
2. PIL.Image.open("data/apple/img_001.jpg")
   → PIL Image, mode="RGB", size=(640, 480)
      ↓
3. .convert("RGB")   → no-op (already RGB), but handles other modes safely
      ↓
4. T.Resize((224, 224))
   → PIL Image, mode="RGB", size=(224, 224)
   → uses bilinear interpolation by default
   → aspect ratio NOT preserved (squashes/stretches to square)
      ↓
5. T.RandomHorizontalFlip(p=0.5)   [TRAINING ONLY]
   → with 50% probability: mirrors the image left-right
      ↓
6. T.RandomRotation(degrees=15)    [TRAINING ONLY]
   → rotates by angle in [-15°, +15°], fills edges with black
      ↓
7. T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)  [TRAINING ONLY]
   → random brightness/contrast/saturation/hue variation
      ↓
8. T.ToTensor()
   → converts PIL Image [H, W, C] uint8 [0,255]
      to torch.Tensor [C, H, W] float32 [0.0, 1.0]
   → shape: (3, 224, 224)
      ↓
9. T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   → output[c] = (input[c] - mean[c]) / std[c]   for c in {R, G, B}
   → values are now roughly in [-2.5, +2.5]
   → shape: (3, 224, 224), dtype: float32
      ↓
10. DataLoader stacks B images:
    → shape: (B, 3, 224, 224), on CPU in pinned memory
      ↓
11. .to(device, non_blocking=True)
    → transferred to GPU (async), shape: (B, 3, 224, 224)
      ↓
12. model(images)
    → feature_extractor → (B, 2048, 1, 1)
    → classifier        → (B, 30) logits
```

### What `.convert("RGB")` handles

| Source mode | What `.convert("RGB")` does |
|-------------|----------------------------|
| `L` (grayscale) | Duplicates single channel to R=G=B |
| `RGBA` | Drops alpha channel, keeps R, G, B |
| `P` (palette) | Converts palette indices to full RGB values |
| `LA` (grayscale+alpha) | Converts grayscale to RGB, drops alpha |
| `RGB` (already) | No-op |
| `CMYK` | Converts print colour space to screen RGB |

---

## 13. The Model Architecture — Layer by Layer

### Tensor dimensions through the network

For a batch of size B=32 and input `(32, 3, 224, 224)`:

```
Input:                                  (32, 3,    224, 224)

Conv2d(3→64, k=7, s=2, p=3):           (32, 64,   112, 112)
BatchNorm2d(64):                        (32, 64,   112, 112)
ReLU:                                   (32, 64,   112, 112)
MaxPool2d(k=3, s=2, p=1):              (32, 64,    56,  56)

Layer1 — 3 × Bottleneck(64→256):       (32, 256,   56,  56)
Layer2 — 4 × Bottleneck(256→512):      (32, 512,   28,  28)
Layer3 — 6 × Bottleneck(512→1024):     (32, 1024,  14,  14)
Layer4 — 3 × Bottleneck(1024→2048):    (32, 2048,   7,   7)

AdaptiveAvgPool2d((1,1)):              (32, 2048,   1,   1)

— classifier head —
Flatten():                             (32, 2048)
Linear(2048→512):                      (32, 512)
BatchNorm1d(512):                      (32, 512)
ReLU:                                  (32, 512)
Dropout(p=0.4):                        (32, 512)   [zeros 40% during train]
Linear(512→30):                        (32, 30)    ← final logits
```

### ResNet Bottleneck block (used in Layers 1–4)

Each Bottleneck block has a **residual shortcut**:

```
Input x
  ↓
Conv1×1 → BN → ReLU    (reduces channels: e.g., 256 → 64)
  ↓
Conv3×3 → BN → ReLU    (same channels: 64 → 64)
  ↓
Conv1×1 → BN           (expands channels back: 64 → 256)
  ↓
(+) Add shortcut connection (x, possibly projected with 1×1 conv if dimensions differ)
  ↓
ReLU
  ↓
Output
```

The shortcut (skip connection) allows gradients to flow directly from loss to early layers without passing through the bottleneck. This is what makes 50-layer networks trainable without vanishing gradients.

### Why AdaptiveAvgPool2d(1,1)?

After Layer4, the feature map is `(32, 2048, 7, 7)`. Instead of flattening to `32 × 2048 × 7 × 7 = 100,352` features, `AdaptiveAvgPool2d(1, 1)` averages each of the 2048 feature maps spatially, producing `(32, 2048, 1, 1)`. This:
- Reduces parameters massively (no huge FC layer)
- Makes the model input-size independent (any spatial size works)
- Performs a form of spatial pooling that's position-invariant

---

## 14. The Training Loop — Step by Step

### Epoch structure

```
For epoch 1 to 30:
│
├── TRAINING PHASE (model.train())
│   For each mini-batch from train_loader:
│   ├── Skip if batch is None (all corrupted)
│   ├── Move images, labels to GPU
│   ├── Forward pass: logits = model(images)        [B, 30]
│   ├── Compute loss: loss = CrossEntropy(logits, labels)  [scalar]
│   ├── zero_grad()
│   ├── loss.backward()                              [compute gradients]
│   ├── optimizer.step()                             [update weights]
│   └── Accumulate loss and accuracy stats
│
├── VALIDATION PHASE (model.eval(), torch.no_grad())
│   For each mini-batch from val_loader:
│   ├── Move images, labels to GPU
│   ├── Forward pass only (no backward)
│   └── Accumulate loss and accuracy stats
│
├── scheduler.step()                                 [decay LR]
├── Print epoch summary
└── If val_acc >= best_val_acc: save checkpoint
```

### Why training and validation are separate phases

During training, `model.train()`:
- Dropout randomly zeros neurons — making the model robust
- BatchNorm uses batch statistics — training-set specific

During validation, `model.eval()`:
- Dropout is disabled — all neurons active, consistent predictions
- BatchNorm uses running statistics — accumulated from training data
- `torch.no_grad()` — no gradient tape, faster and less memory

If you evaluated on the validation set with `model.train()` still active, results would be noisy (dropout) and metrics would be unreliable.

### CrossEntropyLoss formula

For a single sample with true class `c` and logits `z_0, z_1, ..., z_29`:

```
p_i = exp(z_i) / sum_j(exp(z_j))    ← softmax
loss = -log(p_c)                      ← negative log-likelihood of correct class
```

With `label_smoothing=0.1`, the target is softened to `y_c = 0.9` and `y_j = 0.00345` for `j ≠ c`:

```
loss = -sum_j(y_j * log(p_j))
```

---

## 15. The Inference Pipeline — Step by Step

```
1. Call: classify_crops("data/apple/img01.jpg")
      ↓
2. Input normalised to list: ["data/apple/img01.jpg"]
      ↓
3. _get_model() called:
   - First call: loads model from disk, calls model.eval(), caches
   - Subsequent calls: returns cached model instantly
      ↓
4. For each path, _load_image(path):
   - Checks extension
   - PIL.open().convert("RGB")
   - Apply _INFER_TRANSFORM: Resize → ToTensor → Normalize
   - unsqueeze(0) → shape (1, 3, 224, 224)
      ↓
5. torch.cat(tensors, dim=0) → (B, 3, 224, 224)
   .to(device) → moves to GPU if available
      ↓
6. with torch.no_grad():
   logits = model(batch)            → (B, 30) raw scores
   probs  = F.softmax(logits, dim=1) → (B, 30) probabilities, each row sums to 1
   confs, preds = probs.max(dim=1)  → (B,) confidence scores, (B,) class indices
      ↓
7. preds.cpu().tolist() → [0, 5, 27, ...]   (integer class indices)
   confs.cpu().tolist() → [0.93, 0.87, ...]  (confidence scores)
      ↓
8. label_map_inv[0] = "apple", label_map_inv[5] = "grape", etc.
      ↓
9. Return: [{"path": ..., "label": "apple", "label_idx": 0, "confidence": 0.9312}]
```

---

## 16. EDA — All Four Analyses Explained

### Running EDA

```bash
python dataset.py
```

Or programmatically:

```python
from dataset import run_full_eda
import config
run_full_eda(config.DATA_DIR)
```

### Output files produced

| File | What to look for |
|------|-----------------|
| `eda_class_distribution.png` | All bars roughly equal height → balanced dataset. Very short bars → that class is underrepresented and the model may perform poorly on it. |
| `eda_image_sizes.png` | Points clustered near the origin → images are small, upsampling to 224×224 may hurt quality. Points widely spread → high size variation, the resizing step is important. |
| `eda_formats.png` | Mostly one format is fine. If many formats, confirm all are in `SUPPORTED_EXTENSIONS`. |
| `eda_samples.png` | Visual sanity check. Mislabelled images, blank images, or incorrectly placed files will be visible here. |

### What to do if you find problems

- **Class imbalance**: add more images to underrepresented classes, or use `WeightedRandomSampler`
- **Very small images** (e.g., 64×64): consider reducing `resize_x/y` to 64 or 128 to avoid over-upsampling
- **Corrupted images**: delete them from `data/` — they generate `[WARNING]` messages but are safely skipped
- **Mislabelled images**: move them to the correct class directory

---

## 17. Every Hyperparameter — What It Is, Why That Value

| Parameter | File | Default | Effect of Increasing | Effect of Decreasing |
|-----------|------|---------|---------------------|---------------------|
| `resize_x` / `resize_y` | config | 224 | More detail, more memory, slower | Less detail, less memory, faster |
| `batch_size` | config | 32 | Smoother gradients, more memory | Noisier gradients, less memory |
| `epochs` | config | 30 | More training time, risk of overfitting | Less training, risk of underfitting |
| `learning_rate` | config | 1e-4 | Faster learning, risk of divergence | Slower learning, more stable |
| `weight_decay` | config | 1e-4 | More regularisation, smaller weights | Less regularisation |
| `dropout_rate` | config | 0.4 | More regularisation, slower convergence | Less regularisation, faster convergence |
| `lr_min` | config | 1e-6 | Higher final LR, coarser final updates | Lower final LR, finer final updates |
| `RANDOM_FLIP_P` | config | 0.5 | More augmentation diversity | Less augmentation |
| `RANDOM_ROTATION` | config | 15 | More rotation diversity | Less rotation diversity |
| `COLOR_JITTER brightness/contrast/saturation` | config | 0.2 | More colour diversity | Less colour diversity |
| `label_smoothing` | train.py | 0.1 | Less overconfidence, more regularisation | Harder targets, more confidence |
| `NUM_WORKERS` | config | 4 | Faster data loading | Slower data loading |

### How to tune for your specific dataset

- **Overfitting** (train_acc >> val_acc): increase `dropout_rate`, increase `weight_decay`, increase augmentation parameters, reduce `epochs`
- **Underfitting** (both train_acc and val_acc are low): increase `epochs`, increase `learning_rate`, reduce `dropout_rate`, use `pretrained=True`
- **CUDA OOM error**: reduce `batch_size`, reduce `resize_x/y`, reduce `NUM_WORKERS`
- **Slow training**: increase `batch_size`, increase `NUM_WORKERS`, reduce `resize_x/y`

---

## 18. The 30 Crop Classes and Their Integer Labels

The integer label is determined by the **alphabetical sort position** of the class directory name.

| Int | Class | Int | Class | Int | Class |
|-----|-------|-----|-------|-----|-------|
| 0 | apple | 10 | lettuce | 20 | potato |
| 1 | banana | 11 | mango | 21 | raspberry |
| 2 | blueberry | 12 | melon | 22 | rice |
| 3 | cherry | 13 | orange | 23 | soybean |
| 4 | corn | 14 | papaya | 24 | spinach |
| 5 | grape | 15 | peach | 25 | strawberry |
| 6 | grapefruit | 16 | pear | 26 | sugarcane |
| 7 | guava | 17 | pepper | 27 | tomato |
| 8 | kiwi | 18 | pineapple | 28 | watermelon |
| 9 | lemon | 19 | pomegranate | 29 | wheat |

**Important**: These integers are NOT hardcoded anywhere in the code. They are always computed by `build_label_map()` at runtime. If you add a class `"avocado"`, it will be inserted at index 0 and all other labels will shift by 1. Always retrain from scratch after adding/removing/renaming classes.

---

## 19. Checkpoint Format — What Is Saved and Loaded

### File: `checkpoints/final_weights.pth`

A Python pickle file (despite the `.pth` extension, which is a PyTorch convention) containing:

```python
{
    "epoch": int,            # e.g., 22 — which epoch achieved best val_acc
    "val_acc": float,        # e.g., 0.8743 — the best val accuracy
    "model_state_dict": OrderedDict {
        "feature_extractor.0.weight": Tensor(...),
        "feature_extractor.1.weight": Tensor(...),
        "feature_extractor.1.bias":   Tensor(...),
        ... (all ResNet-50 parameters)
        "classifier.1.weight": Tensor([512, 2048]),
        "classifier.1.bias":   Tensor([512]),
        "classifier.2.weight": Tensor([512]),
        "classifier.2.bias":   Tensor([512]),
        "classifier.5.weight": Tensor([30, 512]),
        "classifier.5.bias":   Tensor([30]),
    },
    "optimizer_state_dict": dict {
        "state": {param_id: {"step": int, "exp_avg": Tensor, "exp_avg_sq": Tensor}, ...},
        "param_groups": [{...lr, weight_decay, etc...}]
    }
}
```

### Loading weights for inference only

```python
from model import build_model
from train import load_checkpoint

model = build_model(pretrained=False)
checkpoint = load_checkpoint(model, "checkpoints/final_weights.pth", device="cpu")
print(f"Loaded from epoch {checkpoint['epoch']}, val_acc = {checkpoint['val_acc']:.4f}")
model.eval()
```

### Loading weights to resume training

```python
from model import build_model
from train import load_checkpoint
import torch
import config

model = build_model(pretrained=False)
checkpoint = load_checkpoint(model, "checkpoints/final_weights.pth")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

---

## 20. Grading Script Compatibility — Full Reference

The grading script imports from `interface.py`. Here is the exact contract for each exported name.

### `TheModel` — class

```python
from interface import TheModel

# Create with defaults (reads from config.py)
model = TheModel()

# Create with explicit arguments
model = TheModel(num_classes=30, pretrained=True, dropout_rate=0.4)

# Forward pass
logits = model(torch.randn(4, 3, 224, 224))
# → tensor of shape (4, 30), raw logits
```

### `the_trainer` — function

```python
from interface import the_trainer

history = the_trainer(
    model,           # TheModel instance
    num_epochs,      # int, e.g. 30
    train_loader,    # DataLoader from the_dataloader()
    loss_fn,         # e.g. nn.CrossEntropyLoss()
    optimizer,       # e.g. torch.optim.AdamW(model.parameters(), lr=1e-4)
)

# Optional keyword arguments:
history = the_trainer(
    model, num_epochs, train_loader, loss_fn, optimizer,
    val_loader=val_loader,
    scheduler=scheduler,
    device="cuda",
    save_path="checkpoints/final_weights.pth",
)

# Returns dict:
# history["train_loss"] → list of floats
# history["train_acc"]  → list of floats
# history["val_loss"]   → list of floats
# history["val_acc"]    → list of floats
```

### `the_predictor` — function

```python
from interface import the_predictor

# Single image
results = the_predictor("data/apple/img01.jpg")
# → [{"path": "...", "label": "apple", "label_idx": 0, "confidence": 0.93}]

# Multiple images
results = the_predictor(["img1.jpg", "img2.jpg"])
# → [{"path": ..., "label": ..., ...}, {"path": ..., ...}]

# Access prediction
print(results[0]["label"])      # "apple"
print(results[0]["confidence"]) # 0.9312
print(results[0]["label_idx"])  # 0
```

### `TheDataset` — class

```python
from interface import TheDataset

dataset = TheDataset(root_dir="data/", split="train")
# split can be "train", "val", or "test"

print(len(dataset))                 # number of samples in this split
img_tensor, label = dataset[0]      # tuple: (tensor [3, 224, 224], int)
print(dataset.classes)              # ["apple", "banana", ...]
print(dataset.get_label_name(5))    # "grape"
```

### `the_dataloader` — function

```python
from interface import the_dataloader

loader = the_dataloader("data/", split="train")
# → torch.utils.data.DataLoader

for images, labels in loader:
    # images: tensor (B, 3, 224, 224)
    # labels: tensor (B,) of integers
    pass
```

### `the_batch_size` — int

```python
from interface import the_batch_size
print(the_batch_size)   # 32
```

### `total_epochs` — int

```python
from interface import total_epochs
print(total_epochs)     # 30
```

---

## 21. Running the Project — Every Command

### 1. First-time setup

```bash
cd project_student_name
pip install -r requirements.txt
python generate_sample_data.py
```

### 2. Replace placeholder images

Put real crop images into `data/<class_name>/` folders. Images can be:
- Any resolution (will be resized to 224×224)
- Any format in `SUPPORTED_EXTENSIONS`
- Filename doesn't matter

Aim for at least 50–100 images per class for meaningful training; 500+ per class is ideal.

### 3. Run EDA (optional but recommended)

```bash
python dataset.py
```

Produces: `eda_class_distribution.png`, `eda_image_sizes.png`, `eda_formats.png`, `eda_samples.png`

### 4. Train the model

```bash
python train.py
```

Output (one line per epoch):
```
[train] Using device: cuda
[Epoch 001/030] Train Loss: 2.4321  Acc: 0.2415 | Val Loss: 2.1100  Acc: 0.3200 | LR: 1.00e-04  (45.2s)
  ↳ New best val_acc=0.3200  checkpoint saved → checkpoints/final_weights.pth
[Epoch 002/030] Train Loss: 1.9877  Acc: 0.3541 | Val Loss: 1.7732  Acc: 0.4100 | LR: 9.98e-05  (44.8s)
  ↳ New best val_acc=0.4100  checkpoint saved → checkpoints/final_weights.pth
...
[train] Training complete. Best val_acc = 0.8743
```

### 5. Predict on new images

```bash
# Single image via CLI
python predict.py data/apple/img01.jpg

# Multiple images via CLI
python predict.py data/apple/img01.jpg data/corn/img05.jpg data/wheat/img03.jpg
```

Expected output:
```
data/apple/img01.jpg  →  apple  (conf: 0.9312)
data/corn/img05.jpg   →  corn   (conf: 0.8751)
data/wheat/img03.jpg  →  wheat  (conf: 0.7643)
```

### 6. Predict programmatically

```python
from predict import classify_crops, classify_crops_topk

# Standard prediction
results = classify_crops(["data/apple/img01.jpg", "data/mango/img02.jpg"])
for r in results:
    print(f"{r['label']} ({r['confidence']*100:.1f}%)")

# Top-5 predictions
topk = classify_crops_topk("data/apple/img01.jpg", k=5)
for label, prob in zip(topk[0]["top_labels"], topk[0]["top_probs"]):
    print(f"  {label:20s}: {prob:.4f}")
```

### 7. Evaluate on test set

```python
from model import build_model
from train import load_checkpoint, _evaluate
from dataset import the_dataloader, build_label_map
import torch.nn as nn
import config

model = build_model(pretrained=False)
load_checkpoint(model, config.WEIGHTS_PATH)
label_map   = build_label_map(config.DATA_DIR)
test_loader = the_dataloader(config.DATA_DIR, split="test", label_map=label_map)
loss, acc   = _evaluate(model, test_loader, nn.CrossEntropyLoss(), device="cpu")
print(f"Test Loss: {loss:.4f}   Test Accuracy: {acc:.4f}")
```

### 8. Plot training history

```python
import matplotlib.pyplot as plt
from train import run_training

history = run_training()

epochs = range(1, len(history["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
ax1.plot(epochs, history["val_loss"],   label="Val",   marker="o", markersize=3)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss Curves"); ax1.legend()

ax2.plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
ax2.plot(epochs, history["val_acc"],   label="Val",   marker="o", markersize=3)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy Curves"); ax2.legend()

plt.tight_layout()
plt.savefig("training_history.png", dpi=120)
plt.show()
```

### 9. Sanity-check the model architecture

```bash
python model.py
```

Prints the full model architecture, a test forward pass with dummy input, and the total trainable parameter count.

---

## 22. Common Customisations

### Change input image size

In `config.py`:
```python
resize_x = 256
resize_y = 256
```

No other file needs to change — all transforms and model checks read from `config.py`.  
**Warning**: after changing the input size, delete `final_weights.pth` and retrain. The saved weights are compatible with any input size (due to `AdaptiveAvgPool2d`), but the normalisation statistics and learned filters are optimised for 224×224.

### Change number of classes

In `config.py`:
```python
num_classes = 15    # if you only have 15 classes
```

In `generate_sample_data.py`, update `CROP_CLASSES` to list your actual classes.  
**Always retrain from scratch** after changing `num_classes` — the final `Linear(512, 30)` layer is incompatible with saved weights for a different number of classes.

### Disable augmentation for debugging

In `config.py`:
```python
AUGMENT_TRAIN = False
```

Both the `CropDataset` constructor and `the_dataloader` factory read this flag.

### Freeze the ResNet backbone (train head only)

In `model.py`, add after the backbone is constructed:
```python
for param in self.feature_extractor.parameters():
    param.requires_grad = False
```

Reduces trainable parameters from ~23.5M to ~525K. Trains ~10× faster. Useful when you have very little data (< 100 images per class) to avoid overfitting.

### Use a different backbone

Replace in `model.py`:
```python
from torchvision.models import ResNet18_Weights
backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
```

`in_features = backbone.fc.in_features` will now return `512` (not `2048`), and the `Linear(in_features, 512)` line will create `Linear(512, 512)` — which works fine. No other changes needed because `in_features` is read dynamically.

### Use Windows (fix multiprocessing error)

In `config.py`:
```python
NUM_WORKERS = 0
```

### Reduce memory usage (small GPU)

In `config.py`:
```python
batch_size = 16    # was 32
resize_x   = 128   # was 224
resize_y   = 128   # was 224
```

---

## 23. Troubleshooting — Every Error

### `FileNotFoundError: checkpoints/final_weights.pth`

**Cause**: `predict.py` is trying to load the trained model, but `train.py` has not been run yet (or training crashed before saving the first checkpoint).  
**Fix**: Run `python train.py` and wait for at least one epoch to complete.

### `RuntimeError: Error(s) in loading state_dict`

**Cause**: The saved checkpoint was created with a different model architecture (e.g., different `num_classes`).  
**Fix**: Delete `checkpoints/final_weights.pth` and retrain. The architecture and checkpoint must match.

### `CUDA out of memory`

**Cause**: The batch of images exceeds GPU memory.  
**Fix**: In `config.py`, reduce `batch_size` (try 16, then 8), or reduce `resize_x/resize_y` (try 128).

### `RuntimeError: DataLoader worker (pid ...) is killed by signal`

**Cause**: Multiprocessing issues, common on Windows.  
**Fix**: In `config.py`, set `NUM_WORKERS = 0`.

### `PIL.UnidentifiedImageError: cannot identify image file`

**Cause**: A file with a supported extension (e.g., `.jpg`) is not actually a valid image (corrupted download, wrong extension, placeholder text file).  
**Behaviour**: This is handled gracefully. You will see a `[WARNING]` print. The image is skipped.  
**Fix** (optional): Manually inspect and delete the reported file from `data/`.

### All training predictions are the same class / accuracy stuck at ~3.3%

**Cause**: The model is guessing randomly (1/30 ≈ 3.3%). Possible reasons:
- Using placeholder images (not real crop images) — placeholder images have no learnable patterns
- Severe class imbalance and model always predicts the majority class
- Learning rate is too high or too low
- Data is in wrong format (all images loading as black/corrupted)

**Fix**: Confirm you have replaced placeholder images with real crop images. Run `eda_sample_visualization()` to visually inspect what images are being loaded.

### `ModuleNotFoundError: No module named 'torchvision'`

```bash
pip install torchvision
```

### `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`

**Cause**: Python version < 3.10. The `X | Y` union type syntax requires Python 3.10+.  
**Fix**: Upgrade to Python 3.10 or replace `X | Y` with `Optional[X]` from `typing`.

### `AttributeError: 'NoneType' object has no attribute 'split'`

**Cause**: `classify_crops` returned `None` somehow, or the results list was not indexed correctly.  
**Fix**: Check that the image path passed to `classify_crops` exists and is a valid image.

### Training loss decreases but validation loss increases (overfitting)

**Signs**: `train_acc` continues to rise but `val_acc` plateaus or drops.  
**Fixes**:
- Increase `dropout_rate` (e.g., to `0.5`)
- Increase `weight_decay` (e.g., to `1e-3`)
- Increase augmentation strength (higher `RANDOM_ROTATION`, `COLOR_JITTER_PARAMS`)
- Add more real training images
- Reduce `epochs` and use the checkpoint at the best `val_acc` epoch (already done automatically)

---

## 24. Key Design Decisions and Why

### Why `config.py` has no project imports

`config.py` imports only `os` from the standard library. If it imported from `model.py` or `dataset.py`, those modules in turn import `config.py`, creating a **circular import** that Python cannot resolve. The design rule is: `config.py` is the root of the import tree — everything imports from it, it imports from nothing.

### Why all samples are pooled before splitting (not split per-class)

The split happens after pooling all samples and shuffling. An alternative would be to split each class separately (stratified split). The pooled approach was chosen for simplicity. With 30 classes, if some classes have very few images, stratified splitting would be more principled — you can implement it by calling `_apply_split` per class in `_load_samples`.

### Why the checkpoint uses `>=` not `>` for saving

```python
if val_acc >= best_val_acc:
```

On epoch 1, `best_val_acc = 0.0`, so any non-zero accuracy triggers a save. This ensures a checkpoint always exists after the first epoch. Using strict `>` would also work in practice, but `>=` is safer — it also handles the edge case where two epochs achieve identical accuracy (the later epoch's checkpoint is saved, which has a smaller LR and may generalise slightly better).

### Why `predict.py` uses a module-level cache (`_model_cache`)

Loading a ResNet-50 model from disk takes 1–3 seconds (reading ~100MB from disk). If `classify_crops()` is called in a loop for 1000 images, reloading the model each time would add 1000–3000 seconds of overhead. The cache ensures the load happens once per Python session regardless of how many times the predictor is called.

### Why `the_dataloader` is a function, not a class

A function (factory) is simpler to call and more flexible than a class for this use case. The caller specifies `split=` and gets back a fully configured DataLoader — no need to instantiate a class or manage state. The `CropDataset` class handles the statefulness (the sample list, the label map, the transform pipeline).

### Why the custom head uses `Linear → BatchNorm → ReLU → Dropout` ordering

This is the **canonical ordering** in modern deep learning for fully-connected classification heads. BatchNorm after Linear (not after ReLU) ensures the normalised values pass through ReLU, preserving the zero-centering. Dropout after ReLU (not before) prevents the non-linearity from undoing the zeroing effect.

### Why `_safe_collate` returns `None` for empty batches

Returning `None` is a sentinel that the training loop can check for with `if batch is None: continue`. An alternative would be to skip all samples and not call `collate_fn` at all, but PyTorch's DataLoader always calls it. Another alternative would be to return an empty tensor, but then `images.size(0) == 0` and various matrix operations would fail. `None` + `continue` is the cleanest approach.

### Why `F.softmax` in `predict.py` but not in `model.py`

**In training**: `CrossEntropyLoss` applies `log_softmax` internally. If the model applied `softmax` before, the loss would compute `log(softmax(softmax(...)))`, which is mathematically wrong and numerically less stable.

**In inference**: We want probabilities (sum to 1), not logits (arbitrary scale). Applying `F.softmax` explicitly converts logits to probabilities. The model stays "softmax-free" so it works correctly with both use cases.

---

*End of guide. Every function, variable, design decision, and usage pattern documented.*
