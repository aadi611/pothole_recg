import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path

class PotholeDatasetDownloader:
    """
    Downloads and prepares pothole detection datasets from various sources
    """
    
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset URLs (publicly available pothole datasets)
        self.datasets = {
            "pothole_dataset_v8": {
                "url": "https://universe.roboflow.com/brad-dwyer/pothole-detection-dataset/dataset/2/download/yolov8",
                "type": "roboflow",
                "format": "yolo"
            }
        }
    
    def download_file(self, url, filename):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
    
    def create_sample_dataset(self):
        """
        Creates a sample dataset structure for training
        This method creates placeholder data structure and downloads sample images
        """
        print("Creating sample dataset structure...")
        
        # Create directory structure
        dirs = [
            self.base_dir / "raw",
            self.base_dir / "processed" / "train" / "pothole",
            self.base_dir / "processed" / "train" / "normal",
            self.base_dir / "processed" / "val" / "pothole",
            self.base_dir / "processed" / "val" / "normal",
            self.base_dir / "processed" / "test" / "pothole",
            self.base_dir / "processed" / "test" / "normal"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Download sample images from web (public domain/free images)
        sample_urls = {
            "pothole": [
                "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400",  # Road damage
                "https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?w=400",  # Cracked road
            ],
            "normal": [
                "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=400",  # Normal road
                "https://images.unsplash.com/photo-1570125909232-eb263c188f7e?w=400",  # Highway
            ]
        }
        
        print("Downloading sample images...")
        for category, urls in sample_urls.items():
            for i, url in enumerate(urls):
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        filename = self.base_dir / "raw" / f"{category}_{i+1}.jpg"
                        with open(filename, 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
        
        print("Sample dataset structure created successfully!")
        return True
    
    def download_kaggle_dataset(self, dataset_name):
        """
        Download dataset from Kaggle (requires API key)
        Popular pothole datasets on Kaggle:
        - 'andrewmvd/pothole-detection-dataset'
        - 'sachinpatel21/pothole-image-dataset' 
        """
        try:
            import kaggle
            
            download_path = self.base_dir / "raw" / dataset_name.replace('/', '_')
            download_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading {dataset_name} from Kaggle...")
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(download_path), 
                unzip=True
            )
            print(f"Dataset downloaded to {download_path}")
            return True
            
        except Exception as e:
            print(f"Failed to download from Kaggle: {e}")
            print("Make sure you have kaggle API configured")
            return False
    
    def prepare_dataset_info(self):
        """
        Provides information about available datasets and manual download instructions
        """
        info = """
        POTHOLE DETECTION DATASETS - DOWNLOAD INSTRUCTIONS:
        
        1. ROBOFLOW DATASETS (Recommended):
           - Visit: https://universe.roboflow.com/search?q=pothole
           - Search for "pothole detection" 
           - Choose a dataset with good annotations
           - Download in YOLO or COCO format
           
        2. KAGGLE DATASETS:
           - andrewmvd/pothole-detection-dataset (Object detection format)
           - sachinpatel21/pothole-image-dataset (Classification format)
           - chitholian/annotated-potholes-dataset
           
        3. GITHUB REPOSITORIES:
           - https://github.com/potholedetection/pothole-detection-dataset
           - https://github.com/niennte/pothole-detection
           
        4. MANUAL COLLECTION:
           - Use Google Images with proper licensing
           - Capture your own images with phone/camera
           - Use dashcam footage for real-world scenarios
           
        For this demo, we'll create a sample structure and you can add your images later.
        """
        
        print(info)
        
        # Save to file
        with open(self.base_dir / "dataset_sources.txt", "w") as f:
            f.write(info)
        
        return info

if __name__ == "__main__":
    downloader = PotholeDatasetDownloader()
    
    print("=== POTHOLE DATASET PREPARATION ===")
    print("Choose an option:")
    print("1. Create sample dataset structure (recommended for getting started)")
    print("2. Get dataset download information")
    print("3. Download from Kaggle (requires API setup)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        downloader.create_sample_dataset()
        print("\nSample dataset created! You can now add your own pothole images.")
        print("Add images to:")
        print(f"- {downloader.base_dir}/raw/ (raw images)")
        print("Run the training script after adding images.")
        
    elif choice == "2":
        downloader.prepare_dataset_info()
        
    elif choice == "3":
        dataset = input("Enter Kaggle dataset name (e.g., 'andrewmvd/pothole-detection-dataset'): ")
        downloader.download_kaggle_dataset(dataset)
    
    else:
        print("Invalid choice. Run again with valid option.")