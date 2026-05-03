import torch
import os
from config import WEIGHTS_PATH, CHECKPOINT_DIR


# save checkpoint
checkpoint = {
    "epoch": epoch,  # last epoch number
    "val_acc": val_acc,  # your validation accuracy
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}

torch.save(checkpoint, WEIGHTS_PATH)

print(f"✅ Checkpoint saved at: {WEIGHTS_PATH}")