import sys
import os
import argparse
import shutil
import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn 
from torch import optim
import numpy as np
from tqdm import tqdm
from IPython import get_ipython
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter

# Attempt to import torchvision and handle error if not installed
try:
    from torchvision.utils import make_grid
    from torchvision import transforms
except ImportError as e:
    print("Error: torchvision is not installed. Please install it using 'pip install torchvision'")
    sys.exit(1)

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('weights', exist_ok=True)
os.makedirs('data', exist_ok=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Import custom modules - with error handling
try:
    from src.data import dataset
    from src.data import prepare  # Make sure the error in prepare.py is fixed!
    from src.models.discriminator import Discriminator
    from src.models.generator import Generator
    from src.utils import utils
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure the project structure is correct with all required files.")
    sys.exit(1)

# Optional wandb import with error handling
try:
    import wandb
    wandb_available = True
except ImportError:
    print("Warning: wandb is not installed. Running without wandb integration.")
    wandb_available = False

parser = argparse.ArgumentParser(description='######DImage-to-Image Translation with Conditional Adversarial Nets########')
parser.add_argument('--wandbkey', metavar='wandbkey', default=None,
                    help='Key for Weight and Biases Integration')
parser.add_argument('--projectname', metavar='projectname', default="Cartoonify",
                    help='Project name for Weight and Biases Integration')
parser.add_argument('--wandbentity', metavar='wandbentity',
                    help='Entity for Weight and Biases Integration')
parser.add_argument('--tensorboard', metavar='tensorboard', type=bool, default=True,
                    help='Tensorboard Integration')
parser.add_argument('--kaggle_user', default=None,
                    help="Kaggle API creds required to download Kaggle dataset")
parser.add_argument('--kaggle_key', default=None,
                    help="Kaggle API creds required to download Kaggle dataset")
parser.add_argument('--batch_size', metavar='batch_size', type=int, default=32,
                    help="Batch size")
parser.add_argument('--epoch', metavar='epoch', type=int, default=5,
                    help="Number of epochs")
parser.add_argument('--load_checkpoints', metavar='load_checkpoints', default=False,
                    help="Load model checkpoints")
args = parser.parse_args()

# Safely remove logs directory if it exists
if os.path.isdir("logs"):
    try:
        shutil.rmtree("logs")
    except Exception as e:
        print(f"Warning: Could not remove logs directory: {e}")

os.makedirs("logs", exist_ok=True)
os.makedirs("logs/real", exist_ok=True)
os.makedirs("logs/fake", exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS   = args.epoch
IMG_DIM      = 512
lr           = 2e-4
BATCH_SIZE   = args.batch_size
MAPS_GEN     = 64
MAPS_DISC    = 64
IMG_CHANNELS = 3
L1_LAMBDA    = 100

GEN_CHECKPOINT = '{}_Generator.pt'.format(args.projectname)
DISC_CHECKPOINT = '{}Discriminator.pt'.format(args.projectname)

# Initialize wandb properly
wandb_integration = False
if args.wandbkey and wandb_available:
    try:
        wandb.login(key=args.wandbkey)
        wandb.init(project=args.projectname, entity=args.wandbentity, resume=True)
        print(f"Wandb initialized: {wandb.run.name}")
        wandb_integration = True
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        print("Running without wandb integration.")

# Downloading the dataset - with error handling
try:
    prepare.Download_Dataset(out_path='data')
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please ensure the dataset is available or download it manually to the 'data' directory.")

# Transforms
Transforms = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Data Loaders - with error handling
try:
    train_dataset = dataset.CartoonDataset(datadir='data', transforms=Transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Loading Generator
generator = Generator(img_channels=IMG_CHANNELS, features=MAPS_GEN).to(DEVICE)
if wandb_integration and args.load_checkpoints:
    try:
        checkpoint_path = os.path.join(wandb.run.dir, GEN_CHECKPOINT)
        if os.path.isfile(checkpoint_path):
            generator = torch.load(wandb.restore(GEN_CHECKPOINT).name)
    except Exception as e:
        print(f"Could not load generator checkpoint: {e}")

# Loading Discriminator
discriminator = Discriminator(img_channels=IMG_CHANNELS, features=MAPS_DISC).to(DEVICE)
if wandb_integration and args.load_checkpoints:
    try:
        checkpoint_path = os.path.join(wandb.run.dir, DISC_CHECKPOINT)
        if os.path.isfile(checkpoint_path):
            discriminator = torch.load(wandb.restore(DISC_CHECKPOINT).name)
    except Exception as e:
        print(f"Could not load discriminator checkpoint: {e}")

# Initialize weights
utils.initialize_weights(generator)
utils.initialize_weights(discriminator)

# Loss and Optimizers
gen_optim = optim.Adam(params=generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(params=discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
BCE = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

# Tensorboard Implementation
from torch.utils.tensorboard import SummaryWriter
writer_real = SummaryWriter("logs/real")
writer_fake = SummaryWriter("logs/fake")

if wandb_integration:
    wandb.watch(generator)
    wandb.watch(discriminator)

# Code for COLLAB TENSORBOARD VIEW
try:
    get_ipython().magic("%load_ext tensorboard")
    get_ipython().magic("%tensorboard --logdir logs")
except Exception:
    pass

# Training
discriminator.train()
generator.train()
step = 0
images = []

print(f"Starting training on {DEVICE} for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}")

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    try:
        tqdm_iter = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (face_image, comic_image) in tqdm_iter:
            face_image, comic_image = face_image.to(DEVICE), comic_image.to(DEVICE)

            # Train Discriminator
            fake_image = generator(face_image)
            disc_real = discriminator(face_image, comic_image)
            disc_fake = discriminator(face_image, fake_image)
            disc_real_loss = BCE(disc_real, torch.ones_like(disc_real))
            disc_fake_loss = BCE(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2

            discriminator.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            # Train Generator
            fake_image = generator(face_image)
            disc_fake = discriminator(face_image, fake_image)
            gen_fake_loss = BCE(disc_fake, torch.ones_like(disc_fake))
            L1 = l1_loss(fake_image, comic_image) * L1_LAMBDA
            gen_loss = gen_fake_loss + L1

            generator.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            tqdm_iter.set_postfix(
                D_real=torch.sigmoid(disc_real).mean().item(),
                D_fake=torch.sigmoid(disc_fake).mean().item(),
                disc_loss=disc_loss.item(),
                gen_loss=gen_loss.item()
            )

            if batch_idx % 100 == 0:
                try:
                    torch.save(generator.state_dict(), os.path.join("weights", GEN_CHECKPOINT))
                    torch.save(discriminator.state_dict(), os.path.join("weights", DISC_CHECKPOINT))
                except Exception as e:
                    print(f"Error saving checkpoints: {e}")

                fake_image = fake_image * 0.5 + 0.5
                face_image = face_image * 0.5 + 0.5

                if args.tensorboard:
                    img_grid_real = make_grid(face_image[:8], normalize=True)
                    img_grid_fake = make_grid(fake_image[:8], normalize=True)
                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    step += 1
                    images.append(img_grid_fake.cpu().detach().numpy())

                if wandb_integration:
                    wandb.log({"Discriminator Loss": disc_loss.item(), "Generator Loss": gen_loss.item()})
                    wandb.log({"img": [wandb.Image(img_grid_fake, caption=str(step))]})
                    try:
                        torch.save(generator.state_dict(), os.path.join(wandb.run.dir, GEN_CHECKPOINT))
                        torch.save(discriminator.state_dict(), os.path.join(wandb.run.dir, DISC_CHECKPOINT))
                    except Exception as e:
                        print(f"Error saving checkpoints to wandb: {e}")
    except Exception as e:
        print(f"Error during training: {e}")
        continue

# Create animation from training samples
try:
    if len(images) > 0:
        print("Creating animation from training samples...")
        matplotlib.rcParams['animation.embed_limit'] = 2**64
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        ims = []
        for j, i in tqdm(enumerate(images)):
            ims.append([plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)])
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        f = f"animation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
        ani.save(f, writer=PillowWriter(fps=20))
        print(f"Animation saved as {f}")
        if wandb_integration:
            try:
                ani.save(os.path.join(wandb.run.dir, f), writer=PillowWriter(fps=20))
                print("Animation also saved to wandb")
            except Exception as e:
                print(f"Error saving animation to wandb: {e}")
except Exception as e:
    print(f"Error creating animation: {e}")

print("Training complete!")
