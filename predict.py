# =============================================================================
# predict.py
# Inference utilities for the Agricultural Crop Classifier.
#
# Features:
#   - classify_crops() accepts a single path OR a list of paths
#   - Handles all supported image formats (delegates to PIL)
#   - Safely skips unreadable / corrupted images
#   - Loads weights from config.WEIGHTS_PATH automatically
#   - Returns human-readable class names (not integers)
# =============================================================================

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


# ---------------------------------------------------------------------------
# Transform (same as eval transform in dataset.py — no augmentation)
# ---------------------------------------------------------------------------

_INFER_TRANSFORM = T.Compose([
    T.Resize((config.resize_y, config.resize_x)),
    T.ToTensor(),
    T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
])


# ---------------------------------------------------------------------------
# Model singleton — loaded once, reused across calls
# ---------------------------------------------------------------------------

_model_cache: dict = {}   # { weights_path: (model, label_map_inv, device) }


def _get_model(
    weights_path: str | None = None,
    data_dir:     str | None = None,
    device:       str | None = None,
):
    """
    Loads (or returns cached) model + inverse label map.
    Thread-safety is intentionally left to the caller for simplicity.
    """
    weights_path = weights_path or config.WEIGHTS_PATH
    data_dir     = data_dir     or config.DATA_DIR

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if weights_path not in _model_cache:
        model = build_model(pretrained=False)
        load_checkpoint(model, path=weights_path, device=device)
        model.to(device)
        model.eval()

        label_map     = build_label_map(data_dir)
        label_map_inv = {v: k for k, v in label_map.items()}

        _model_cache[weights_path] = (model, label_map_inv, device)
        print(f"[predict] Model loaded from '{weights_path}' on {device}")

    return _model_cache[weights_path]


# ---------------------------------------------------------------------------
# Image loading helper
# ---------------------------------------------------------------------------

def _load_image(img_path: str) -> torch.Tensor | None:
    """
    Opens one image file, applies inference transforms, and returns a
    [1, C, H, W] tensor — or None if the file is corrupted / unreadable.
    """
    ext = Path(img_path).suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        print(f"[predict] Unsupported format skipped: {img_path}")
        return None
    try:
        image  = Image.open(img_path).convert("RGB")
        tensor = _INFER_TRANSFORM(image)          # [C, H, W]
        return tensor.unsqueeze(0)                # [1, C, H, W]
    except (UnidentifiedImageError, OSError, Exception) as exc:
        print(f"[predict] Could not read '{img_path}': {exc}")
        return None


# ---------------------------------------------------------------------------
# Core prediction function  (exposed via interface.py as the_predictor)
# ---------------------------------------------------------------------------

def classify_crops(
    img_paths:    Union[str, list[str]],
    weights_path: str | None = None,
    data_dir:     str | None = None,
    device:       str | None = None,
    return_probs: bool = False,
) -> list[dict]:
    """
    Runs inference on one or more image files.

    Parameters
    ----------
    img_paths    : a single path string OR a list of path strings.
                   Each path should point to a file inside the data/ directory
                   (or any accessible path on disk).
    weights_path : path to .pth checkpoint (default: config.WEIGHTS_PATH)
    data_dir     : dataset root used to reconstruct the label map
    device       : "cuda" | "cpu" (auto-detected if None)
    return_probs : if True, includes the full probability vector in output

    Returns
    -------
    A list of dicts (one per input path) with keys:
        "path"       : original input path
        "label"      : predicted class name (str)
        "label_idx"  : predicted class index (int)
        "confidence" : probability of the predicted class (float)
        "probs"      : (optional) full probability list over all classes
    """
    # Normalise input to a list
    if isinstance(img_paths, (str, Path)):
        img_paths = [str(img_paths)]
    else:
        img_paths = [str(p) for p in img_paths]

    model, label_map_inv, _device = _get_model(weights_path, data_dir, device)
    if device is not None:
        _device = device

    results = []

    # ── Batch processing ──────────────────────────────────────────────────
    tensors_ok  = []   # valid tensors
    paths_ok    = []   # corresponding paths
    paths_error = []   # paths that failed to load

    for path in img_paths:
        t = _load_image(path)
        if t is not None:
            tensors_ok.append(t)
            paths_ok.append(path)
        else:
            paths_error.append(path)

    # Sentinel results for unreadable images
    for path in paths_error:
        results.append({
            "path":       path,
            "label":      "ERROR",
            "label_idx":  -1,
            "confidence": 0.0,
        })

    if tensors_ok:
        batch  = torch.cat(tensors_ok, dim=0).to(_device)   # [B, C, H, W]

        with torch.no_grad():
            logits  = model(batch)                           # [B, num_classes]
            probs   = F.softmax(logits, dim=1)               # [B, num_classes]
            confs, preds = probs.max(dim=1)                  # [B], [B]

        for path, pred, conf, prob_vec in zip(
            paths_ok,
            preds.cpu().tolist(),
            confs.cpu().tolist(),
            probs.cpu().tolist(),
        ):
            entry = {
                "path":       path,
                "label":      label_map_inv.get(pred, f"class_{pred}"),
                "label_idx":  pred,
                "confidence": round(conf, 4),
            }
            if return_probs:
                entry["probs"] = prob_vec
            results.append(entry)

    # Preserve original order
    order = {p: i for i, p in enumerate(img_paths)}
    results.sort(key=lambda r: order.get(r["path"], 0))

    return results


# ---------------------------------------------------------------------------
# Top-k variant (useful for inspection / grading reports)
# ---------------------------------------------------------------------------

def classify_crops_topk(
    img_paths:    Union[str, list[str]],
    k:            int  = 5,
    weights_path: str | None = None,
    data_dir:     str | None = None,
    device:       str | None = None,
) -> list[dict]:
    """
    Returns top-*k* predictions for each image.

    Each result dict has keys:
        "path", "top_labels" (list[str]), "top_probs" (list[float])
    """
    full = classify_crops(
        img_paths, weights_path=weights_path,
        data_dir=data_dir, device=device, return_probs=True,
    )
    model, label_map_inv, _ = _get_model(weights_path, data_dir, device)

    out = []
    for r in full:
        if r["label"] == "ERROR":
            out.append({"path": r["path"], "top_labels": [], "top_probs": []})
            continue
        probs = r.get("probs", [])
        top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
        out.append({
            "path":       r["path"],
            "top_labels": [label_map_inv.get(i, f"class_{i}") for i in top_indices],
            "top_probs":  [round(probs[i], 4) for i in top_indices],
        })
    return out


# ---------------------------------------------------------------------------
# Quick CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python predict.py path/to/img1.jpg path/to/img2.png ...")
    else:
        predictions = classify_crops(paths)
        for p in predictions:
            print(f"{p['path']}  →  {p['label']}  (conf: {p['confidence']:.4f})")
