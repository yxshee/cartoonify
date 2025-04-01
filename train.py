"""
Training script for the Cartoonify GAN model.

This script trains a GAN (Generative Adversarial Network) to transform
regular face images into cartoon-style images. It uses a conditional GAN
architecture with a U-Net generator and a PatchGAN discriminator.

Usage:
    python train.py [--wandbkey KEY] [--projectname NAME] [--wandbentity ENTITY]
                    [--tensorboard BOOL] [--batch_size SIZE] [--epoch NUM]
                    [--load_checkpoints BOOL]
"""
import datetime
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import sys
import wandb
import argparse
import shutil
import os
from tqdm import tqdm 
from IPython import get_ipython

import numpy as np
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

# Fix for macOS OpenMP runtime issue
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set up command line argument parser
parser = argparse.ArgumentParser(description='######Image-to-Image Translation with Conditional Adversarial Nets########')

# Weights & Biases integration parameters
parser.add_argument('--wandbkey', metavar='wandbkey', default=None,
                    help='API Key for Weights & Biases integration')

parser.add_argument('--projectname', metavar='projectname', default="Cartoonify",
                    help='Project name for Weights & Biases')

parser.add_argument('--wandbentity', metavar='wandbentity',
                    help='Team/organization name for Weights & Biases')

# Tensorboard integration parameter
parser.add_argument('--tensorboard', metavar='tensorboard', type=bool, default=True,
                    help='Enable/disable Tensorboard logging')

# Kaggle credentials for dataset access
parser.add_argument('--kaggle_user', default=None,
                    help="Kaggle username required to download dataset")

parser.add_argument('--kaggle_key', default=None,
                    help="Kaggle API key required to download dataset")

# Training parameters
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=32,
                    help="Batch size for training")
                    
parser.add_argument('--epoch', metavar='epoch', type=int, default=5,
                    help="Number of training epochs")

# Checkpoint parameters
parser.add_argument('--load_checkpoints', metavar='load_checkpoints', default=False,
                    help="Whether to load existing model checkpoints")
                    
args = parser.parse_args()

# Remove existing logs directory if it exists
shutil.rmtree("logs") if os.path.isdir("logs") else ""

# Configuration for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
NUM_EPOCHS   = args.epoch            # Number of training epochs
IMG_DIM      = 512                   # Image dimensions (height and width)
lr           = 2e-4                  # Learning rate for Adam optimizer
BATCH_SIZE   = args.batch_size       # Number of images per batch
MAPS_GEN     = 64                    # Number of feature maps in generator
MAPS_DISC    = 64                    # Number of feature maps in discriminator
IMG_CHANNELS = 3                     # RGB image channels
L1_LAMBDA    = 100                   # Weight for L1 loss term in generator loss

# Checkpoint file names
GEN_CHECKPOINT = '{}_Generator.pt'.format(args.projectname)
DISC_CHECKPOINT = '{}Discriminator.pt'.format(args.projectname)

# Download the dataset 
prepare.Download_Dataset(out_path='data')  # Custom module that manages dataset download

# Define image transformations for preprocessing
Trasforms = transforms.Compose([
    transforms.Resize(IMG_DIM),          # Resize images to target dimension
    transforms.CenterCrop(IMG_DIM),      # Center crop to ensure square images
    transforms.ToTensor(),               # Convert PIL image to PyTorch tensor
    transforms.Normalize(                # Normalize to range [-1, 1]
        (0.5, 0.5, 0.5),                 # Mean for each channel
        (0.5, 0.5, 0.5))                 # Standard deviation for each channel
    ])

# Load the dataset and create data loaders
train_dataset = dataset.CartoonDataset(datadir='data', transforms=Trasforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Create directory for saving model weights
os.makedirs("weights", exist_ok=True)

# Configure Weights & Biases integration if API key is provided
if args.wandbkey:
    wandb_integration = True
    wandb.login(key=args.wandbkey)
    wandb.init(project=args.projectname, entity=args.wandbentity, resume=True)
    print(wandb.run.name)
else:
    wandb_integration = False
    print("Running without wandb integration")

# Initialize or load Generator model
if wandb_integration and os.path.isdir(os.path.join(wandb.run.dir, GEN_CHECKPOINT)) and args.load_checkpoints:
    # Load from W&B if available
    generator = torch.load(wandb.restore(GEN_CHECKPOINT).name)
else:
    # Check if model exists in local weights directory
    if os.path.exists(os.path.join("weights", GEN_CHECKPOINT)) and args.load_checkpoints:
        generator = torch.load(os.path.join("weights", GEN_CHECKPOINT))
    else:
        # Create new generator model
        generator = Generator(img_channels=IMG_CHANNELS, features=MAPS_GEN).to(DEVICE)

# Initialize or load Discriminator model
if wandb_integration and os.path.isdir(os.path.join(wandb.run.dir, DISC_CHECKPOINT)) and args.load_checkpoints:
    # Load from W&B if available
    discriminator = torch.load(wandb.restore(DISC_CHECKPOINT).name)
else:
    # Check if model exists in local weights directory
    if os.path.exists(os.path.join("weights", DISC_CHECKPOINT)) and args.load_checkpoints:
        discriminator = torch.load(os.path.join("weights", DISC_CHECKPOINT))
    else:
        # Create new discriminator model
        discriminator = Discriminator(img_channels=IMG_CHANNELS, features=MAPS_DISC).to(DEVICE)

# Initialize model weights with custom initialization
utils.initialize_weights(generator)
utils.initialize_weights(discriminator)

# Define optimizers for generator and discriminator
gen_optim = optim.Adam(params=generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(params=discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Define loss functions
BCE = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits for adversarial loss
l1_loss = nn.L1Loss()         # L1 loss for pixel-wise difference between generated and target

# Initialize TensorBoard writers for visualization
writer_real = SummaryWriter(f"logs/real")  # For real images
writer_fake = SummaryWriter(f"logs/fake")  # For generated images

# Register models with W&B for tracking
if wandb_integration:
    wandb.watch(generator)
    wandb.watch(discriminator)

# Set up TensorBoard in Jupyter/Colab if applicable
try:
    get_ipython().magic("%load_ext tensorboard")
    get_ipython().magic("%tensorboard --logdir logs")
except:
    pass

# Set models to training mode
discriminator.train()
generator.train()
step = 0
images = []  # Store generated images for animation

# Training loop
for epoch in range(1, NUM_EPOCHS+1):
    # Create progress bar for current epoch
    tqdm_iter = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for batch_idx, (face_image, comic_image) in tqdm_iter:
        # Move input tensors to device (GPU/CPU)
        face_image, comic_image = face_image.to(DEVICE), comic_image.to(DEVICE)
        
        # ---------- Train Discriminator ----------
        # Generate fake cartoon images
        fake_image = generator(face_image)
        
        # Get discriminator predictions for real and fake pairs
        disc_real = discriminator(face_image, comic_image)  # For real pairs
        disc_fake = discriminator(face_image, fake_image)   # For fake pairs

        # Calculate discriminator losses
        disc_real_loss = BCE(disc_real, torch.ones_like(disc_real))   # Real pairs should be classified as 1
        disc_fake_loss = BCE(disc_fake, torch.zeros_like(disc_fake))  # Fake pairs should be classified as 0

        # Total discriminator loss is average of real and fake losses
        disc_loss = (disc_real_loss + disc_fake_loss) / 2

        # Update discriminator weights
        discriminator.zero_grad()
        disc_loss.backward()
        disc_optim.step()

        # ---------- Train Generator ----------
        fake_image = generator(face_image)  # Generate fake images again
        disc_fake = discriminator(face_image, fake_image)  # Get discriminator's prediction for fake images
        gen_fake_loss = BCE(disc_fake, torch.ones_like(disc_fake))  # Generator wants discriminator to classify fake as real
        L1 = l1_loss(fake_image, comic_image) * L1_LAMBDA  # L1 loss for pixel-wise difference
        gen_loss = gen_fake_loss + L1  # Total generator loss

        # Update generator weights
        generator.zero_grad()
        gen_loss.backward()
        gen_optim.step()
        
        # Update progress bar with current losses
        tqdm_iter.set_postfix(
            D_real=torch.sigmoid(disc_real).mean().item(),
            D_fake=torch.sigmoid(disc_fake).mean().item(),
            disc_loss=disc_loss.item(),
            gen_loss=gen_loss.item()
        )

        # Save model checkpoints and log images every 100 batches
        if batch_idx % 100 == 0:
            torch.save(generator.state_dict(), os.path.join("weights", GEN_CHECKPOINT))
            torch.save(discriminator.state_dict(), os.path.join("weights", DISC_CHECKPOINT))

            fake_image = fake_image * 0.5 + 0.5  # Denormalize images to range [0, 1]
            face_image = face_image * 0.5 + 0.5  # Denormalize images to range [0, 1]
            
            if args.tensorboard:
                img_grid_real = make_grid(face_image[:8], normalize=True)
                img_grid_fake = make_grid(fake_image[:8], normalize=True)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                step += 1
                images.append(img_grid_fake.cpu().detach().numpy())

            if wandb_integration:
                wandb.log({"Discriminator Loss": disc_loss.item(), "Generator Loss": gen_loss.item()})
                wandb.log({"img": [wandb.Image(img_grid_fake, caption=step)]})

                torch.save(generator.state_dict(), os.path.join(wandb.run.dir, GEN_CHECKPOINT))
                torch.save(discriminator.state_dict(), os.path.join(wandb.run.dir, DISC_CHECKPOINT))

# Create and save animation of generated images
try:
    matplotlib.rcParams['animation.embed_limit'] = 2**64
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = []
    for j, i in tqdm(enumerate(images)):
        ims.append([plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)]) 
        
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
    f = "animation{}.gif".format(datetime.datetime.now()).replace(":", "")

    if wandb_integration:
        ani.save(os.path.join(wandb.run.dir, f), writer=PillowWriter(fps=20)) 
    ani.save(f, writer=PillowWriter(fps=20)) 
except:
    pass
