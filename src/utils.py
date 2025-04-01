import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

def initialize_weights(model):
    """
    Initialize weights for the model layers
    
    Args:
        model: PyTorch model
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

def save_image(img_array, output_path):
    """
    Save a numpy array as an image
    
    Args:
        img_array (numpy.ndarray): Image as a numpy array (H, W, C)
        output_path (str): Path to save the image
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert to uint8 if not already
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Save the image
    Image.fromarray(img_array).save(output_path)

def resize_image(img_array, target_size):
    """
    Resize an image array to target size
    
    Args:
        img_array (numpy.ndarray): Image as numpy array
        target_size (tuple): Target (width, height)
        
    Returns:
        numpy.ndarray: Resized image
    """
    target_width, target_height = target_size
    current_height, current_width = img_array.shape[:2]
    
    # Skip if already the right size
    if current_width == target_width and current_height == target_height:
        return img_array
    
    # PIL expects (width, height) but our array is (height, width, channels)
    resized_img = cv2.resize(img_array, (target_width, target_height), 
                             interpolation=cv2.INTER_LANCZOS4)
    return resized_img

def plot_comparison(original, cartoon, figsize=(10, 5)):
    """
    Plot original image next to cartoonified version
    
    Args:
        original: Original image (either path or numpy array)
        cartoon: Cartoon image (either path or numpy array)
        figsize: Figure size for the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Handle inputs that are file paths
    if isinstance(original, str):
        original = np.array(Image.open(original).convert("RGB"))
    if isinstance(cartoon, str):
        cartoon = np.array(Image.open(cartoon).convert("RGB"))
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Display original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Display cartoon image
    axes[1].imshow(cartoon)
    axes[1].set_title("Cartoonified Image")
    axes[1].axis("off")
    
    plt.tight_layout()
    return fig

def create_gif_from_images(image_paths, output_path, duration=200):
    """
    Create an animated GIF from a list of image paths
    
    Args:
        image_paths (list): List of paths to images
        output_path (str): Path to save the output GIF
        duration (int): Duration of each frame in milliseconds
    """
    images = [Image.open(path) for path in image_paths]
    images[0].save(output_path, save_all=True, append_images=images[1:], 
                   optimize=False, duration=duration, loop=0)
