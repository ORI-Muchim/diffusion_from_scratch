# custom_dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CustomDataset")

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=28, image_channels=1, verbose=False):
        self.root_dir = root_dir
        self.image_size = image_size
        self.image_channels = image_channels
        self.verbose = verbose
        
        # Get all image files
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        self.image_files = []
        
        # Create default transform if none provided
        if transform is None:
            if image_channels == 1:
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda img: self._center_crop_resize(img, vertical_shift=0.2)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            else:  # RGB
                self.transform = transforms.Compose([
                    transforms.Lambda(lambda img: self._center_crop_resize(img, vertical_shift=0.2)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        # Validate the dataset and remove corrupted images
        self._validate_dataset()
        
    def _center_crop_resize(self, img, target_size=None, vertical_shift=0.2):
        """
        Crop image from center with slight upward shift, then resize to target size.
        vertical_shift: positive values move crop up, negative values move down (range: 0-1)
        """
        if target_size is None:
            target_size = (self.image_size, self.image_size)
        
        # Crop to square based on shortest dimension
        width, height = img.size
        
        if width > height:
            # Landscape image
            crop_size = height
            left = (width - crop_size) // 2
            right = left + crop_size
            top = 0
            bottom = crop_size
        else:
            # Portrait image
            crop_size = width
            left = 0
            right = crop_size
            
            # Apply vertical shift
            max_shift = (height - crop_size)
            shift_pixels = int(max_shift * vertical_shift)
            
            top = (height - crop_size) // 2 - shift_pixels
            
            # Check boundaries
            if top < 0:
                top = 0
            if top + crop_size > height:
                top = height - crop_size
                
            bottom = top + crop_size
        
        # Crop with vertical shift
        img_cropped = img.crop((left, top, right, bottom))
        
        # Resize to target size
        return img_cropped.resize(target_size, Image.LANCZOS)

    def _validate_dataset(self):
        logger.info(f"Validating images in {self.root_dir}...")
        total_files = 0
        valid_files = 0
        corrupted_files = 0
        non_image_files = 0
        
        # First, get all potential image files
        all_files = []
        for file in os.listdir(self.root_dir):
            if file.lower().endswith(self.valid_extensions):
                all_files.append(os.path.join(self.root_dir, file))
            elif not file.endswith(".json"):  # Exclude JSON metadata files from warning
                non_image_files += 1
        
        total_files = len(all_files)
        
        # Then, validate each file
        for file_path in all_files:
            try:
                # Try to open the image to check if it's valid
                with Image.open(file_path) as img:
                    # Try to load the image data
                    img.verify()
                    
                    # If the image passes verification, add it to valid files
                    self.image_files.append(file_path)
                    valid_files += 1
                    
                    if self.verbose and valid_files % 100 == 0:
                        logger.info(f"Validated {valid_files} images...")
                        
            except (OSError, UnidentifiedImageError) as e:
                corrupted_files += 1
                logger.warning(f"Skipping corrupted image: {file_path}. Error: {str(e)}")
                
                # Optionally rename corrupted files
                try:
                    corrupted_path = file_path + ".corrupted"
                    os.rename(file_path, corrupted_path)
                    logger.info(f"Renamed corrupted file to {corrupted_path}")
                except:
                    pass
        
        logger.info(f"Dataset validation complete: Found {valid_files} valid images, {corrupted_files} corrupted images, {non_image_files} non-image files.")
        
        if valid_files == 0:
            raise ValueError(f"No valid images found in {self.root_dir}. Please check your dataset.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Use error handling to deal with problematic images at runtime
        max_retries = 5
        current_idx = idx
        
        for attempt in range(max_retries):
            try:
                img_path = self.image_files[current_idx]
                
                # Load image with careful error handling
                image = Image.open(img_path)
                
                # Convert to the proper mode
                image = image.convert('RGB' if self.image_channels == 3 else 'L')
                
                # Apply transformations
                if self.transform:
                    image = self.transform(image)
                
                # Return image and a dummy label (0) since we don't need labels for diffusion
                return image, 0
                
            except (OSError, UnidentifiedImageError) as e:
                logger.warning(f"Error loading image at index {current_idx} (attempt {attempt+1}/{max_retries}): {str(e)}")
                
                # Try the next image
                current_idx = (current_idx + 1) % len(self.image_files)
                
                # If we've tried too many times, something is seriously wrong
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load any valid image after {max_retries} attempts.")
                    
                    # Return a random noise tensor as a last resort
                    if self.image_channels == 1:
                        return torch.randn(1, self.image_size, self.image_size), 0
                    else:
                        return torch.randn(3, self.image_size, self.image_size), 0
