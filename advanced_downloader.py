"""
Advanced Pothole Dataset Downloader with Real Internet Sources
This script helps download real pothole datasets from various sources
"""

import os
import requests
import zipfile
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

class AdvancedPotholeDownloader:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def download_from_url(self, url, filename):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
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
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def download_roboflow_sample(self):
        """
        Download sample images from Roboflow Universe
        Note: For full datasets, you need to register at roboflow.com
        """
        print("Downloading sample images from Roboflow...")
        
        # These are sample URLs - for full datasets, you need Roboflow API key
        sample_images = {
            "pothole_1.jpg": "https://source.roboflow.com/pothole1.jpg",
            "pothole_2.jpg": "https://source.roboflow.com/pothole2.jpg", 
            "normal_1.jpg": "https://source.roboflow.com/road1.jpg",
            "normal_2.jpg": "https://source.roboflow.com/road2.jpg"
        }
        
        raw_dir = self.base_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        success_count = 0
        for filename, url in sample_images.items():
            filepath = raw_dir / filename
            if self.download_from_url(url, filepath):
                success_count += 1
        
        print(f"Downloaded {success_count} sample images")
        return success_count > 0
    
    def download_github_datasets(self):
        """
        Download datasets from GitHub repositories
        """
        print("Accessing GitHub datasets...")
        
        github_datasets = [
            {
                "name": "Pothole Detection Dataset",
                "zip_url": "https://github.com/chitholian/Pothole-Detection-Dataset/archive/refs/heads/main.zip",
                "description": "Annotated pothole dataset with bounding boxes"
            },
            {
                "name": "Road Damage Dataset", 
                "info": "Visit: https://github.com/sekilab/RoadDamageDetector",
                "description": "Large dataset of road damages including potholes"
            }
        ]
        
        for dataset in github_datasets:
            print(f"\n--- {dataset['name']} ---")
            print(f"Description: {dataset['description']}")
            
            if 'zip_url' in dataset:
                print(f"Downloading from: {dataset['zip_url']}")
                zip_path = self.base_dir / f"{dataset['name'].replace(' ', '_')}.zip"
                
                if self.download_from_url(dataset['zip_url'], zip_path):
                    print("Downloaded successfully!")
                    # Extract if it's a zip file
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            extract_path = self.base_dir / dataset['name'].replace(' ', '_')
                            zip_ref.extractall(extract_path)
                        print(f"Extracted to: {extract_path}")
                        return True
                    except Exception as e:
                        print(f"Error extracting: {e}")
                        
            elif 'info' in dataset:
                print(dataset['info'])
                
        return False
    
    def setup_kaggle_download(self):
        """
        Setup and download from Kaggle
        """
        print("=== KAGGLE DATASET DOWNLOAD ===")
        print("To download from Kaggle, you need to:")
        print("1. Create account at kaggle.com")
        print("2. Go to Account -> API -> Create New Token")
        print("3. Place kaggle.json in ~/.kaggle/ or current directory")
        print()
        
        # Check if kaggle is available
        try:
            import kaggle
            print("✅ Kaggle API is available!")
            
            recommended_datasets = [
                "andrewmvd/pothole-detection-dataset",
                "sachinpatel21/pothole-image-dataset", 
                "chitholian/annotated-potholes-dataset",
                "sumaiyatasmeem/pothole-image-dataset"
            ]
            
            print("\nRecommended Kaggle datasets:")
            for i, dataset in enumerate(recommended_datasets, 1):
                print(f"{i}. {dataset}")
            
            choice = input("\nEnter dataset number to download (1-4), or 0 to skip: ")
            
            if choice in ['1', '2', '3', '4']:
                dataset_name = recommended_datasets[int(choice) - 1]
                return self.download_kaggle_dataset(dataset_name)
                
        except ImportError:
            print("❌ Kaggle package not installed. Install with: pip install kaggle")
            
        return False
    
    def download_kaggle_dataset(self, dataset_name):
        """Download specific Kaggle dataset"""
        try:
            import kaggle
            
            download_path = self.base_dir / "kaggle_data" / dataset_name.replace('/', '_')
            download_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading {dataset_name}...")
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(download_path),
                unzip=True
            )
            
            print(f"✅ Dataset downloaded to: {download_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading from Kaggle: {e}")
            if "401" in str(e):
                print("Make sure your Kaggle API token is properly configured")
            return False
    
    def download_open_images(self):
        """
        Download road/pothole images from Open Images Dataset
        """
        print("=== OPEN IMAGES DATASET ===")
        print("Downloading road damage images from Google's Open Images...")
        
        # Open Images has a command line tool
        print("To download from Open Images:")
        print("1. Install: pip install openimages")
        print("2. Use command: openimages download --classes 'Road,Pothole' --limit 1000")
        print("3. Or visit: https://storage.googleapis.com/openimages/web/index.html")
        
        return False
    
    def create_download_script(self):
        """
        Create a batch script for easy dataset downloading
        """
        script_content = """
# Pothole Dataset Download Commands

## Method 1: Kaggle (Best option)
# 1. Setup Kaggle API: pip install kaggle
# 2. Get API token from kaggle.com/account
# 3. Run commands:

kaggle datasets download -d andrewmvd/pothole-detection-dataset
kaggle datasets download -d sachinpatel21/pothole-image-dataset

## Method 2: Manual Downloads
# Visit these URLs and download manually:

# 1. Roboflow Universe
https://universe.roboflow.com/brad-dwyer/pothole-detection-dataset
https://universe.roboflow.com/viren-dhanwani/pothole-detection-computer-vision-project

# 2. GitHub Repositories  
https://github.com/chitholian/Pothole-Detection-Dataset
https://github.com/sekilab/RoadDamageDetector

# 3. Academic Datasets
https://data.mendeley.com/datasets/5ty2wb6gvg/1  # Pothole Image Dataset
https://www.crcv.ucf.edu/data/LISA/    # Traffic Sign Detection (has road images)

## Method 3: Public Image Sources
# Use these for additional training data:
# - Unsplash (free high-quality images)
# - Pixabay (free stock images)  
# - Government transportation datasets

## After downloading:
# 1. Extract all archives
# 2. Organize images into data/raw/pothole/ and data/raw/normal/
# 3. Run: python data_preprocessing.py
"""
        
        script_path = self.base_dir / "download_instructions.md"
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        print(f"✅ Download instructions saved to: {script_path}")
        return script_path

def main():
    downloader = AdvancedPotholeDownloader()
    
    print("🛣️ ADVANCED POTHOLE DATASET DOWNLOADER 🕳️")
    print("=" * 50)
    
    options = [
        ("1", "Setup Kaggle download (Recommended)", downloader.setup_kaggle_download),
        ("2", "Download from GitHub repositories", downloader.download_github_datasets), 
        ("3", "Get download instructions for all sources", downloader.create_download_script),
        ("4", "Download sample images (for testing)", downloader.download_roboflow_sample),
        ("5", "Show Open Images info", downloader.download_open_images)
    ]
    
    print("\nChoose download method:")
    for code, desc, _ in options:
        print(f"{code}. {desc}")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    # Find and execute the chosen option
    for code, desc, func in options:
        if choice == code:
            print(f"\n--- {desc} ---")
            success = func()
            if success:
                print("\n✅ Operation completed successfully!")
                print("Next steps:")
                print("1. Organize downloaded images into data/raw/pothole/ and data/raw/normal/")
                print("2. Run: python data_preprocessing.py")
                print("3. Run: python train_model.py")
            else:
                print("\n⚠️ Some issues occurred. Check the instructions above.")
            return
    
    print("Invalid choice. Please run again.")

if __name__ == "__main__":
    main()