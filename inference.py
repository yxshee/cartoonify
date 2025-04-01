import torch 
from torchvision import transforms
import argparse
from PIL import Image
import os
import numpy as np

# Import the model architecture
from src.models.generator import Generator

parser = argparse.ArgumentParser()

parser.add_argument('--img', metavar='img', type=str,
                    help="Image to Cartoonize")
parser.add_argument('--output', metavar='output', type=str, default="cartoon_output.jpg",
                    help="Output image path")
parser.add_argument('--model', metavar='model', type=str, default="weights/Cartoonify_Generator.pt",
                    help="Path to model weights")

args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_DIM = 512
MAPS_GEN = 64
IMG_CHANNELS = 3

# Create transforms
transforms_list = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))
])

# Load the model
try:
    # First try loading the state_dict
    model = Generator(img_channels=IMG_CHANNELS, features=MAPS_GEN).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
except:
    # If that fails, try loading the entire model
    try:
        model = torch.load(args.model, map_location=DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you've trained the model first.")
        exit(1)

model.eval()

# Process the image
if not os.path.exists(args.img):
    print(f"Image file {args.img} not found")
    exit(1)

# Open and process the image
img = Image.open(args.img).convert("RGB")
input_tensor = transforms_list(img).unsqueeze(0).to(DEVICE)

# Generate the cartoon image
with torch.no_grad():
    output = model(input_tensor)
    
# Convert the output tensor to an image
output = output.squeeze()
output = output * 0.5 + 0.5  # Denormalize
output = output.clamp(0, 1)
output = output.cpu().permute(1, 2, 0).numpy() * 255
output = output.astype(np.uint8)

# Save the output image
output_img = Image.fromarray(output)
output_img.save(args.output)
print(f"Cartoon image saved to {args.output}")

# Display a preview if running in an interactive environment
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(img))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("Cartoon Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
except:
    pass
