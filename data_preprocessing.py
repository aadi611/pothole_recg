import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import json
import shutil

class PotholeDataProcessor:
    """
    Processes and prepares pothole detection dataset for training
    """
    
    def __init__(self, data_dir="data", img_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        
        # Create directories
        for split in ['train', 'val', 'test']:
            for category in ['pothole', 'normal']:
                (self.processed_dir / split / category).mkdir(parents=True, exist_ok=True)
    
    def setup_augmentations(self):
        """
        Setup data augmentation pipeline using Albumentations
        """
        self.train_transforms = A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.Blur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_and_process_image(self, image_path, transform=None, save_path=None):
        """
        Load and process a single image
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if transform:
                transformed = transform(image=image)
                image = transformed['image']
            else:
                # Basic resize if no transform
                image = cv2.resize(image, self.img_size)
                image = image / 255.0  # Normalize
            
            # Save processed image if path provided
            if save_path:
                if isinstance(image, np.ndarray) and image.max() <= 1.0:
                    # Denormalize for saving
                    save_img = (image * 255).astype(np.uint8)
                else:
                    save_img = image.astype(np.uint8)
                
                save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_path), save_img_bgr)
            
            return image
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def organize_raw_data(self):
        """
        Organize raw data into pothole and normal categories
        """
        print("Organizing raw data...")
        
        raw_files = list(self.raw_dir.glob("*"))
        if not raw_files:
            print("No raw data found. Please add images to the 'data/raw' directory.")
            return False
        
        # Simple classification based on filename (you can modify this logic)
        for file_path in tqdm(raw_files):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                filename = file_path.name.lower()
                
                # Classify based on filename keywords
                if any(keyword in filename for keyword in ['pothole', 'crack', 'damage', 'hole']):
                    category = 'pothole'
                else:
                    category = 'normal'
                
                # Copy to appropriate category in raw folder
                category_dir = self.raw_dir / category
                category_dir.mkdir(exist_ok=True)
                
                if not (category_dir / file_path.name).exists():
                    shutil.copy2(file_path, category_dir / file_path.name)
        
        return True
    
    def split_and_process_data(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Split data and process with augmentations
        """
        print("Setting up augmentations...")
        self.setup_augmentations()
        
        print("Splitting and processing data...")
        
        # Get all images by category
        categories = ['pothole', 'normal']
        dataset_info = {
            'train': {'pothole': [], 'normal': []},
            'val': {'pothole': [], 'normal': []},
            'test': {'pothole': [], 'normal': []}
        }
        
        for category in categories:
            category_dir = self.raw_dir / category
            if not category_dir.exists():
                print(f"Category directory {category_dir} not found. Creating empty category.")
                category_dir.mkdir(exist_ok=True)
                continue
                
            images = list(category_dir.glob("*"))
            images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            if len(images) == 0:
                print(f"No images found in {category} category")
                continue
            
            # Split data
            train_imgs, temp_imgs = train_test_split(images, test_size=(1-train_ratio), random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42)
            
            # Process and save images
            splits = {
                'train': train_imgs,
                'val': val_imgs, 
                'test': test_imgs
            }
            
            for split, img_list in splits.items():
                print(f"Processing {len(img_list)} {category} images for {split}...")
                
                for i, img_path in enumerate(tqdm(img_list)):
                    # Choose transform based on split
                    transform = self.train_transforms if split == 'train' else self.val_transforms
                    
                    # Process image
                    save_path = self.processed_dir / split / category / f"{img_path.stem}_{i}.jpg"
                    processed_img = self.load_and_process_image(img_path, transform, save_path)
                    
                    if processed_img is not None:
                        dataset_info[split][category].append(str(save_path))
                    
                    # For training set, create additional augmented versions
                    if split == 'train' and processed_img is not None:
                        for aug_idx in range(3):  # Create 3 additional augmented versions
                            aug_save_path = self.processed_dir / split / category / f"{img_path.stem}_{i}_aug_{aug_idx}.jpg"
                            aug_processed = self.load_and_process_image(img_path, self.train_transforms, aug_save_path)
                            if aug_processed is not None:
                                dataset_info[split][category].append(str(aug_save_path))
        
        # Save dataset info
        with open(self.data_dir / 'dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Print statistics
        self.print_dataset_stats(dataset_info)
        
        return dataset_info
    
    def print_dataset_stats(self, dataset_info):
        """Print dataset statistics"""
        print("\n=== DATASET STATISTICS ===")
        for split in ['train', 'val', 'test']:
            pothole_count = len(dataset_info[split]['pothole'])
            normal_count = len(dataset_info[split]['normal'])
            total = pothole_count + normal_count
            
            print(f"{split.upper()}:")
            print(f"  Pothole images: {pothole_count}")
            print(f"  Normal images: {normal_count}")
            print(f"  Total: {total}")
            if total > 0:
                print(f"  Pothole ratio: {pothole_count/total:.2%}")
            print()
    
    def create_sample_images(self):
        """
        Create sample synthetic images for demonstration
        """
        print("Creating sample synthetic images...")
        
        # Create synthetic pothole-like images
        for i in range(10):
            # Normal road image (gray with some texture)
            normal_img = np.random.randint(100, 150, (*self.img_size, 3), dtype=np.uint8)
            # Add road texture
            noise = np.random.normal(0, 10, (*self.img_size, 3))
            normal_img = np.clip(normal_img + noise, 0, 255).astype(np.uint8)
            
            normal_path = self.raw_dir / 'normal' / f'synthetic_normal_{i}.jpg'
            normal_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(normal_path), normal_img)
            
            # Pothole image (darker with irregular shapes)
            pothole_img = normal_img.copy()
            # Add dark irregular shape (simulating pothole)
            center = (np.random.randint(50, self.img_size[0]-50), np.random.randint(50, self.img_size[1]-50))
            axes = (np.random.randint(20, 40), np.random.randint(20, 40))
            cv2.ellipse(pothole_img, center, axes, 0, 0, 360, (50, 50, 50), -1)
            # Add some roughness around edges
            cv2.circle(pothole_img, center, np.random.randint(45, 55), (70, 70, 70), -1)
            
            pothole_path = self.raw_dir / 'pothole' / f'synthetic_pothole_{i}.jpg'
            pothole_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pothole_path), pothole_img)
        
        print("Sample synthetic images created!")

if __name__ == "__main__":
    processor = PotholeDataProcessor()
    
    print("=== POTHOLE DATA PROCESSING ===")
    print("Choose an option:")
    print("1. Create sample synthetic data and process")
    print("2. Process existing raw data")
    print("3. Organize raw data first")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        processor.create_sample_images()
        processor.split_and_process_data()
        print("Sample data created and processed successfully!")
        
    elif choice == "2":
        success = processor.split_and_process_data()
        if success:
            print("Data processed successfully!")
        else:
            print("Please add images to data/raw/ directory first")
            
    elif choice == "3":
        processor.organize_raw_data()
        print("Raw data organized. Now run option 2 to process it.")
        
    else:
        print("Invalid choice. Run again with valid option.")