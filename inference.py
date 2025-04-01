"""
Inference script for the Cartoonify GAN model.

This script applies a pre-trained GAN generator to transform
regular face images into cartoon-style images. It handles loading
the trained model, processing input images, and saving the cartoonified results.

Usage:
    python inference.py --img INPUT_IMAGE --output OUTPUT_IMAGE --model MODEL_PATH
"""
import torch 
from torchvision import transforms
import argparse
from PIL import Image
import os
import numpy as np

# Import the generator model architecture
from src.models.generator import Generator

# Set up command line argument parser
parser = argparse.ArgumentParser(description="Cartoonify Image Transformer")

parser.add_argument('--img', metavar='img', type=str,
                    help="Path to the input image to cartoonize")
parser.add_argument('--output', metavar='output', type=str, default="cartoon_output.jpg",
                    help="Path where the cartoonified image will be saved")
parser.add_argument('--model', metavar='model', type=str, default="weights/Cartoonify_Generator.pt",
                    help="Path to the pre-trained generator model weights")

args = parser.parse_args()

# Set device (GPU if available, otherwise CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_DIM = 512     # Image dimensions for processing
MAPS_GEN = 64     # Feature maps in generator (must match training config)
IMG_CHANNELS = 3  # RGB image channels

# Define image preprocessing transformations
transforms_list = transforms.Compose([
    transforms.Resize(IMG_DIM),          # Resize input image to model's expected size
    transforms.CenterCrop(IMG_DIM),      # Center crop to ensure square images
    transforms.ToTensor(),               # Convert PIL image to tensor
    transforms.Normalize(                # Normalize to range [-1, 1]
        (0.5, 0.5, 0.5),                 # Mean for each channel (same as training)
        (0.5, 0.5, 0.5))                 # Standard deviation (same as training)
])

# Load the pre-trained generator model
try:
    # First try loading just the state_dict (preferred method)
    model = Generator(img_channels=IMG_CHANNELS, features=MAPS_GEN).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
except:
    # If that fails, try loading the entire model object
    try:
        model = torch.load(args.model, map_location=DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you've trained the model first.")
        exit(1)

# Set model to evaluation mode (disables dropout, etc.)
model.eval()

# Verify the input image exists
if not os.path.exists(args.img):
    print(f"Image file {args.img} not found")
    exit(1)

# Open and preprocess the input image
img = Image.open(args.img).convert("RGB")  # Ensure RGB format
input_tensor = transforms_list(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

# Generate the cartoon image
with torch.no_grad():  # Disable gradient computation for inference
    output = model(input_tensor)
    
# Process the output tensor back to a displayable image
output = output.squeeze()           # Remove batch dimension
output = output * 0.5 + 0.5         # Denormalize from [-1,1] to [0,1]
output = output.clamp(0, 1)         # Ensure values stay in valid range
output = output.cpu().permute(1, 2, 0).numpy() * 255  # CHW -> HWC and scale to [0,255]
output = output.astype(np.uint8)    # Convert to 8-bit format for image saving

# Save the cartoonified image
output_img = Image.fromarray(output)
output_img.save(args.output)
print(f"Cartoon image saved to {args.output}")

# Display a preview if running in an interactive environment
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title("Original Image")
    plt.axis('off')
    
    # Show cartoonified image
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("Cartoon Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
except:
    pass  # Skip visualization if matplotlib not available or not in interactive mode
