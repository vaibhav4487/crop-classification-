# =============================================================================
# interface.py
# Standardised interface for the grading script.
#
# ⚠️  THIS FILE MUST ONLY CONTAIN IMPORTS AND ALIASING.
#     Do NOT add any logic, classes, or functions here.
#
# The grading program will call:
#   TheModel        – the model class
#   the_trainer     – function that runs the training loop
#   the_predictor   – function that runs inference on image path(s)
#   TheDataset      – the custom Dataset class
#   the_dataloader  – function that returns a DataLoader
#   the_batch_size  – batch size value (int)
#   total_epochs    – total number of training epochs (int)
# =============================================================================

# Replace CropClassifier with the name of your model class
from model import CropClassifier as TheModel

# Replace train_model with the function inside train.py that runs the training loop
from train import train_model as the_trainer

# Replace classify_crops with the function inside predict.py for inference
from predict import classify_crops as the_predictor

# Replace CropDataset with your custom Dataset class
from dataset import CropDataset as TheDataset

# Replace the_dataloader with your custom DataLoader factory
from dataset import the_dataloader as the_dataloader

# Replace batch_size / epochs with the variable names in config.py
from config import batch_size as the_batch_size
from config import epochs as total_epochs
