import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from src.utils import save_image, resize_image
from src.config import IMG_DIM, DEVICE, MAPS_GEN, IMG_CHANNELS

def load_model(model_path=None):
    """
    Load the pre-trained cartoonify model
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        model: Loaded PyTorch model
    """
    if model_path is None:
        model_path = os.path.join('weights', 'Cartoonify_Generator.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    try:
        model = torch.load(model_path, map_location=DEVICE)
        model.eval()  # Set model to evaluation mode
        model.to(DEVICE)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def prepare_image(image_path, img_dim=IMG_DIM):
    """
    Load and preprocess an image for the model
    
    Args:
        image_path (str): Path to the input image
        img_dim (int): Dimension to resize the image to
        
    Returns:
        tensor: Preprocessed image tensor
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    transform = transforms.Compose([
        transforms.Resize(img_dim),
        transforms.CenterCrop(img_dim),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension
        return img_tensor, img.size
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {e}")

def cartoonify_image(input_path, output_path, model_path=None):
    """
    Transform an image into a cartoon-style image
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the output image
        model_path (str, optional): Path to the model checkpoint
    
    Returns:
        str: Path to the saved cartoonified image
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Prepare image
    img_tensor, original_size = prepare_image(input_path)
    
    # Generate cartoon image
    with torch.no_grad():
        cartoon_tensor = model(img_tensor)
    
    # Convert tensor to image and save
    cartoon_tensor = (cartoon_tensor * 0.5 + 0.5).clamp(0, 1)
    cartoon_image = cartoon_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    cartoon_image = cartoon_image.astype(np.uint8)
    
    # Resize back to original dimensions
    cartoon_image = resize_image(cartoon_image, original_size)
    
    # Save the output
    save_image(cartoon_image, output_path)
    
    return output_path

def process_multiple_images(input_dir, output_dir, model_path=None):
    """
    Process multiple images in a directory
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save output images
        model_path (str, optional): Path to the model checkpoint
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No compatible images found in {input_dir}")
        return
    
    # Load model once for all images
    model = load_model(model_path)
    
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"cartoon_{img_file}")
        
        try:
            # Prepare image
            img_tensor, original_size = prepare_image(input_path)
            
            # Generate cartoon image
            with torch.no_grad():
                cartoon_tensor = model(img_tensor)
            
            # Convert tensor to image and save
            cartoon_tensor = (cartoon_tensor * 0.5 + 0.5).clamp(0, 1)
            cartoon_image = cartoon_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
            cartoon_image = cartoon_image.astype(np.uint8)
            
            # Resize back to original dimensions
            cartoon_image = resize_image(cartoon_image, original_size)
            
            # Save the output
            save_image(cartoon_image, output_path)
            
            print(f"Processed {img_file} -> {output_path}")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Transform photos into cartoon-style images')
    
    # Single image processing
    parser.add_argument('--input', '-i', type=str, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, help='Path to save output image')
    
    # Batch processing
    parser.add_argument('--input_dir', type=str, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, help='Directory to save output images')
    
    # Model
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if args.input and args.output:
        cartoonify_image(args.input, args.output, args.model)
        print(f"Image cartoonified successfully! Saved to {args.output}")
    elif args.input_dir and args.output_dir:
        process_multiple_images(args.input_dir, args.output_dir, args.model)
        print(f"Batch processing complete! Images saved to {args.output_dir}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
