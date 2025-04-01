import os
import torch

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Image parameters
IMG_DIM = 512
IMG_CHANNELS = 3

# Model parameters
MAPS_GEN = 64
MAPS_DISC = 64
L1_LAMBDA = 100
LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_EPOCHS = 10

# File paths
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights')
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, 'sample_images')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')

# Create directories if they don't exist
for directory in [WEIGHTS_DIR, LOGS_DIR, DATA_DIR, SAMPLE_IMAGES_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model file names
GEN_CHECKPOINT = 'Cartoonify_Generator.pt'
DISC_CHECKPOINT = 'Cartoonify_Discriminator.pt'

# Weights & Biases configuration
WANDB_PROJECT_NAME = "Cartoonify"
