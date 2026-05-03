# =============================================================================
# config.py
# Central configuration file for the Agricultural Crop Classification project.
# ALL hyperparameters and path settings are defined here.
# Other modules import from this file — never hardcode values elsewhere.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
WEIGHTS_PATH    = os.path.join(CHECKPOINT_DIR, "final_weights.pth")

# ---------------------------------------------------------------------------
# Image / Input settings
# ---------------------------------------------------------------------------
resize_x        = 224          # target width  (pixels)
resize_y        = 224          # target height (pixels)
input_channels  = 3            # RGB

# ImageNet normalisation — safe default for ResNet-based models
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# Supported image extensions (lower-case)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
num_classes     = 30           # 30 agricultural crop classes
dropout_rate    = 0.4          # dropout applied before final classifier head

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
batch_size      = 32
epochs          = 30
learning_rate   = 1e-4
weight_decay    = 1e-4         # L2 regularisation for AdamW

# Learning-rate scheduler (CosineAnnealingLR)
lr_scheduler_T_max = epochs    # one full cosine cycle over all epochs
lr_min             = 1e-6      # minimum LR at cycle end

# ---------------------------------------------------------------------------
# Data split ratios  (train / val / test)
# ---------------------------------------------------------------------------
TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.10

# ---------------------------------------------------------------------------
# Augmentation (training only)
# ---------------------------------------------------------------------------
AUGMENT_TRAIN      = True
RANDOM_FLIP_P      = 0.5       # probability for horizontal flip
RANDOM_ROTATION    = 15        # degrees
COLOR_JITTER       = True      # brightness / contrast / saturation jitter
COLOR_JITTER_PARAMS = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
NUM_WORKERS     = 4            # DataLoader worker threads
RANDOM_SEED     = 42
DEVICE          = "cuda"       # "cuda" | "cpu" — auto-detected in train.py
