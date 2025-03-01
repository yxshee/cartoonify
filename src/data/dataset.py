import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CartoonDataset(Dataset):
    def __init__(self, datadir, transforms=None):
        self.datadir = datadir
        self.transforms = transforms
        
        # Assuming datadir contains 'face' and 'comic' subdirectories
        self.face_dir = os.path.join(datadir, 'face')
        self.comic_dir = os.path.join(datadir, 'comic')
        
        # Create directories if they don't exist
        os.makedirs(self.face_dir, exist_ok=True)
        os.makedirs(self.comic_dir, exist_ok=True)
        
        # Get list of image files (only consider files that exist in both directories)
        self.face_images = sorted([f for f in os.listdir(self.face_dir) 
                                  if os.path.isfile(os.path.join(self.face_dir, f)) and 
                                  f.endswith(('.jpg', '.jpeg', '.png'))])
        
        self.comic_images = sorted([f for f in os.listdir(self.comic_dir) 
                                   if os.path.isfile(os.path.join(self.comic_dir, f)) and 
                                   f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Use only files that exist in both directories with the same name
        common_files = set(self.face_images).intersection(set(self.comic_images))
        self.face_images = sorted(list(common_files))
        self.comic_images = sorted(list(common_files))
        
        if len(self.face_images) == 0:
            print("No matching image pairs found. Please make sure you have corresponding files in both directories.")
            # Create sample images for testing
            self._create_sample_images()
        
    def _create_sample_images(self):
        # Create sample images for testing if no real data is available
        print("Creating sample images for testing...")
        sample_size = 10
        for i in range(sample_size):
            # Create a random face image (gray)
            face_img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
            face_file = f"sample_{i}.jpg"
            Image.fromarray(face_img).save(os.path.join(self.face_dir, face_file))
            
            # Create a random comic image (colorful)
            comic_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            comic_file = f"sample_{i}.jpg"
            Image.fromarray(comic_img).save(os.path.join(self.comic_dir, comic_file))
        
        self.face_images = [f"sample_{i}.jpg" for i in range(sample_size)]
        self.comic_images = [f"sample_{i}.jpg" for i in range(sample_size)]
        print(f"Created {sample_size} sample image pairs for testing.")

    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, idx):
        face_img_path = os.path.join(self.face_dir, self.face_images[idx])
        comic_img_path = os.path.join(self.comic_dir, self.comic_images[idx])
        
        face_image = Image.open(face_img_path).convert("RGB")
        comic_image = Image.open(comic_img_path).convert("RGB")
        
        if self.transforms:
            face_image = self.transforms(face_image)
            comic_image = self.transforms(comic_image)
        
        return face_image, comic_image

if __name__ == '__main__':
    CartoonDataset()