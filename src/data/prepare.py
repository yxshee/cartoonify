import os
import shutil
import random
from tqdm import tqdm
from PIL import Image
import numpy as np

def Download_Dataset(out_path='data'):
    """
    Downloads or prepares the dataset for training.
    This function either downloads from an online source or creates 
    sample data if download is not possible.
    
    Args:
        out_path (str): Directory to save the dataset
    """
    face_dir = os.path.join(out_path, 'face')
    comic_dir = os.path.join(out_path, 'comic')
    
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(comic_dir, exist_ok=True)
    
    # Check if directories already contain images
    face_files = [f for f in os.listdir(face_dir) 
                 if os.path.isfile(os.path.join(face_dir, f)) and 
                 f.endswith(('.jpg', '.jpeg', '.png'))]
    
    comic_files = [f for f in os.listdir(comic_dir) 
                  if os.path.isfile(os.path.join(comic_dir, f)) and 
                  f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # If files already exist, no need to download/create again
    if len(face_files) > 0 and len(comic_files) > 0:
        print(f"Dataset already exists with {len(face_files)} face images and {len(comic_files)} comic images.")
        return
    
    # Try to download from an online source (could be implemented with requests, kaggle API, etc.)
    try:
        import kaggle
        # Check if kaggle credentials are available
        if os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            print("Attempting to download dataset from Kaggle...")
            # This is a placeholder - replace with actual dataset reference
            # kaggle.api.dataset_download_files('dataset/cartoonify', path=out_path, unzip=True)
            raise NotImplementedError("Kaggle download not implemented yet.")
        else:
            print("Kaggle credentials not found. Creating sample data instead.")
            raise ImportError("Kaggle credentials not available")
    except (ImportError, NotImplementedError):
        print("Creating sample dataset for testing...")
        _create_sample_dataset(face_dir, comic_dir)

def _create_sample_dataset(face_dir, comic_dir, n_samples=100):
    """
    Creates a sample dataset with random images
    
    Args:
        face_dir (str): Directory to save face images
        comic_dir (str): Directory to save comic images
        n_samples (int): Number of sample image pairs to generate
    """
    print(f"Generating {n_samples} sample image pairs...")
    
    for i in tqdm(range(n_samples)):
        # Create a sample face image (grayscale-like)
        img_size = 512
        face_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Add some random shapes to make it look like a face
        # Background
        face_img[:, :] = np.random.randint(180, 220, (3,))
        
        # Face shape
        center_x, center_y = img_size//2, img_size//2
        for y in range(img_size):
            for x in range(img_size):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < img_size//3:
                    face_img[y, x] = np.random.randint(200, 240, (3,))
        
        # Eyes
        eye_radius = img_size//10
        eye_y = center_y - img_size//10
        
        # Left eye
        left_eye_x = center_x - img_size//6
        for y in range(img_size):
            for x in range(img_size):
                dist = np.sqrt((x - left_eye_x)**2 + (y - eye_y)**2)
                if dist < eye_radius:
                    face_img[y, x] = np.random.randint(100, 150, (3,))
        
        # Right eye
        right_eye_x = center_x + img_size//6
        for y in range(img_size):
            for x in range(img_size):
                dist = np.sqrt((x - right_eye_x)**2 + (y - eye_y)**2)
                if dist < eye_radius:
                    face_img[y, x] = np.random.randint(100, 150, (3,))
        
        # Mouth
        mouth_y = center_y + img_size//5
        for y in range(mouth_y-5, mouth_y+5):
            for x in range(center_x-img_size//5, center_x+img_size//5):
                if 0 <= y < img_size and 0 <= x < img_size:
                    face_img[y, x] = np.random.randint(100, 150, (3,))
        
        # Save face image
        face_file = f"sample_{i:04d}.jpg"
        Image.fromarray(face_img).save(os.path.join(face_dir, face_file))
        
        # Create a cartoon version (more colorful and stylized)
        cartoon_img = face_img.copy()
        
        # Make it more vibrant
        for y in range(img_size):
            for x in range(img_size):
                # Add some random color variations
                cartoon_img[y, x] = np.clip(cartoon_img[y, x] * np.random.uniform(0.7, 1.3, (3,)), 0, 255).astype(np.uint8)
        
        # Add some random bright areas for cartoon effect
        for _ in range(20):
            cx, cy = np.random.randint(0, img_size, 2)
            radius = np.random.randint(10, 50)
            color = np.random.randint(180, 255, 3)
            
            for y in range(img_size):
                for x in range(img_size):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < radius:
                        factor = 1 - (dist / radius)
                        cartoon_img[y, x] = np.clip(
                            cartoon_img[y, x] * (1 - factor) + color