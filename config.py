import torch
import os
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cuda"
ROOT = "./dataset"
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR = os.path.join(ROOT, "val")
TEST_DIR = os.path.join(ROOT, "test")
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
NUM_WORKERS = 14

NUM_EPOCHS = 301
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_DISC = "./model/"
CHECKPOINT_GEN = "./model/"

