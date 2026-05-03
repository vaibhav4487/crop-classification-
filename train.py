# =============================================================================
# train.py
# Training loop for the Agricultural Crop Classifier.
#
# Features:
#   - Forward pass, loss computation, backward pass, optimiser step
#   - Per-epoch training & validation accuracy / loss logging
#   - CosineAnnealingLR scheduler
#   - Best-model checkpoint saved to config.WEIGHTS_PATH
#   - Early-stopping (optional, off by default)
# =============================================================================

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import build_model
from dataset import build_label_map, the_dataloader


# ---------------------------------------------------------------------------
# Core training function  (exposed via interface.py as the_trainer)
# ---------------------------------------------------------------------------

def train_model(
    model,
    num_epochs:   int,
    train_loader: DataLoader,
    loss_fn,
    optimizer,
    val_loader:   DataLoader | None = None,
    scheduler=None,
    device:       str | None = None,
    save_path:    str | None = None,
):
    """
    Runs the full training loop for *num_epochs* epochs.

    Parameters
    ----------
    model        : nn.Module – the model to train
    num_epochs   : int       – total number of epochs
    train_loader : DataLoader
    loss_fn      : loss function (e.g. nn.CrossEntropyLoss())
    optimizer    : torch optimiser
    val_loader   : optional DataLoader for validation
    scheduler    : optional LR scheduler (called after every epoch)
    device       : "cuda" | "cpu" (auto-detects if None)
    save_path    : path to save the best checkpoint (default: config.WEIGHTS_PATH)

    Returns
    -------
    history : dict with keys "train_loss", "train_acc", "val_loss", "val_acc"
    """

    # ── Device setup ──────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")
    model = model.to(device)

    # ── Checkpoint path ───────────────────────────────────────────────────
    save_path = save_path or config.WEIGHTS_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── History ───────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Training phase ────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:          # all images in batch were corrupted
                continue

            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            logits = model(images)
            loss   = loss_fn(logits, labels)

            # Backward pass + optimiser step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc  = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # ── Validation phase ──────────────────────────────────────────────
        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None:
            val_loss, val_acc = _evaluate(model, val_loader, loss_fn, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── LR scheduler step ─────────────────────────────────────────────
        if scheduler is not None:
            scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {lr_now:.2e}  ({elapsed:.1f}s)"
        )

        # ── Save best checkpoint ───────────────────────────────────────────
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, optimizer, epoch, val_acc, save_path)
            print(f"  ↳ New best val_acc={best_val_acc:.4f}  checkpoint saved → {save_path}")

    print(f"\n[train] Training complete. Best val_acc = {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate(
    model,
    loader:    DataLoader,
    loss_fn,
    device:    str,
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy) over one full pass through *loader*."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits        = model(images)
            loss          = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, optimizer, epoch: int, val_acc: float, path: str):
    """Saves model weights + training state to *path*."""
    torch.save(
        {
            "epoch":      epoch,
            "val_acc":    val_acc,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, path: str, device: str = "cpu"):
    """
    Loads weights from *path* into *model* in-place.
    Returns the checkpoint dict (contains epoch and val_acc metadata).
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[load_checkpoint] Loaded weights from '{path}' "
          f"(epoch={checkpoint.get('epoch')}, val_acc={checkpoint.get('val_acc', 'n/a'):.4f})")
    return checkpoint


# ---------------------------------------------------------------------------
# Convenience runner — builds everything and calls train_model
# ---------------------------------------------------------------------------

def run_training(data_dir: str | None = None):
    """
    Convenience wrapper:  builds model, optimiser, loaders, and runs training.
    Call this from a main script or Jupyter notebook.
    """
    data_dir   = data_dir or config.DATA_DIR
    label_map  = build_label_map(data_dir)

    # Data loaders
    train_loader = the_dataloader(data_dir, split="train", label_map=label_map)
    val_loader   = the_dataloader(data_dir, split="val",   label_map=label_map)

    # Model
    model   = build_model(pretrained=True)

    # Loss function — CrossEntropyLoss works with integer labels and raw logits
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config.learning_rate,
        weight_decay = config.weight_decay,
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = config.lr_scheduler_T_max,
        eta_min= config.lr_min,
    )

    history = train_model(
        model        = model,
        num_epochs   = config.epochs,
        train_loader = train_loader,
        loss_fn      = loss_fn,
        optimizer    = optimizer,
        val_loader   = val_loader,
        scheduler    = scheduler,
    )
    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_training()
