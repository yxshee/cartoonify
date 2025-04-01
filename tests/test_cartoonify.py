import os
import unittest
import tempfile
import numpy as np
from PIL import Image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cartoonify import prepare_image, cartoonify_image
from src.utils import resize_image, save_image
from src.config import IMG_DIM, DEVICE

class TestCartoonify(unittest.TestCase):
    def setUp(self):
        # Create a temporary test image
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_image_path = os.path.join(self.temp_dir.name, 'test_image.jpg')
        
        # Create a simple test image
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        Image.fromarray(test_img).save(self.test_image_path)
        
    def tearDown(self):
        # Clean up temporary files
        self.temp_dir.cleanup()
    
    def test_prepare_image(self):
        """Test if image preparation works properly"""
        # This test will only run if a test model is available
        try:
            img_tensor, original_size = prepare_image(self.test_image_path)
            
            # Check tensor shape and properties
            self.assertEqual(img_tensor.shape, (1, 3, IMG_DIM, IMG_DIM))
            self.assertEqual(img_tensor.device.type, DEVICE)
            
            # Check original size is preserved
            self.assertEqual(original_size, (100, 100))
        except FileNotFoundError:
            self.skipTest("Model file not found. Skipping test.")
    
    def test_resize_image(self):
        """Test image resizing functionality"""
        # Create a test array
        test_array = np.ones((200, 300, 3), dtype=np.uint8) * 128
        
        # Resize to new dimensions
        resized = resize_image(test_array, (150, 100))
        
        # Check new dimensions
        self.assertEqual(resized.shape, (100, 150, 3))
    
    def test_save_image(self):
        """Test if image saving works properly"""
        # Create a test array
        test_array = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Save to a temporary location
        temp_output = os.path.join(self.temp_dir.name, 'output.jpg')
        save_image(test_array, temp_output)
        
        # Check if file exists and is an image
        self.assertTrue(os.path.exists(temp_output))
        try:
            img = Image.open(temp_output)
            self.assertEqual(img.size, (100, 100))
        except Exception as e:
            self.fail(f"Failed to open saved image: {e}")
    
    def test_cartoonify_integration(self):
        """Integration test for the full cartoonification process"""
        # This test will only run if a model is available
        
        # Skip test if no model is available
        if not os.path.exists(os.path.join('weights', 'Cartoonify_Generator.pt')):
            self.skipTest("Model file not found. Skipping integration test.")
            return
        
        # Create a temp output file
        temp_output = os.path.join(self.temp_dir.name, 'cartoon_output.jpg')
        
        try:
            # Run cartoonification
            result_path = cartoonify_image(self.test_image_path, temp_output)
            
            # Check if output exists
            self.assertTrue(os.path.exists(result_path))
            
            # Verify it's a valid image
            img = Image.open(result_path)
            self.assertEqual(img.size, (100, 100))  # Should maintain original size
        except Exception as e:
            self.fail(f"Cartoonify integration test failed: {e}")

if __name__ == '__main__':
    unittest.main()
