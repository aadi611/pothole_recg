import os
import sys
import subprocess
import json
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def create_directory_structure():
    """Create necessary directories"""
    print("Creating directory structure...")
    
    directories = [
        "data",
        "data/raw",
        "data/raw/pothole", 
        "data/raw/normal",
        "data/processed",
        "data/processed/train",
        "data/processed/train/pothole",
        "data/processed/train/normal", 
        "data/processed/val",
        "data/processed/val/pothole",
        "data/processed/val/normal",
        "data/processed/test",
        "data/processed/test/pothole",
        "data/processed/test/normal",
        "models",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")

def setup_environment():
    """Setup the complete environment"""
    print("=== POTHOLE DETECTION PROJECT SETUP ===")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Create directories
    create_directory_structure()
    
    # Install requirements
    if not install_requirements():
        return False
    
    print("\n=== SETUP COMPLETE ===")
    print("Next steps:")
    print("1. Add pothole and normal road images to data/raw/pothole/ and data/raw/normal/")
    print("2. Run: python data_preprocessing.py")
    print("3. Run: python train_model.py")
    print("\nOr use the quick start option:")
    print("4. Run: python data_preprocessing.py (choose option 1 for synthetic data)")
    print("5. Run: python train_model.py (to train with synthetic data)")
    
    return True

def quick_start_demo():
    """Quick start with synthetic data"""
    print("=== QUICK START DEMO ===")
    
    print("This will create synthetic data and train a demo model.")
    proceed = input("Continue? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("Cancelled.")
        return
    
    try:
        # Import and run data preprocessing
        from data_preprocessing import PotholeDataProcessor
        processor = PotholeDataProcessor()
        
        print("Creating synthetic data...")
        processor.create_sample_images()
        processor.split_and_process_data()
        
        print("Starting model training...")
        from train_model import PotholeDetectionModel
        model = PotholeDetectionModel()
        
        # Train with fewer epochs for demo
        history = model.train_model(epochs=5, batch_size=16, base_model='MobileNetV2')
        
        # Show results
        model.plot_training_history()
        model.evaluate_model()
        
        print("✅ Demo completed successfully!")
        print("Check the 'models' folder for saved models.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all dependencies are installed correctly.")

def main():
    print("=== POTHOLE DETECTION SETUP ===")
    print("1. Full setup (install dependencies and create structure)")
    print("2. Quick start demo (synthetic data + training)")
    print("3. Directory structure only")
    print("4. Install requirements only")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        setup_environment()
        
    elif choice == "2":
        if not Path("requirements.txt").exists():
            print("❌ requirements.txt not found!")
            return
        
        # First setup environment
        if setup_environment():
            # Then run demo
            quick_start_demo()
            
    elif choice == "3":
        create_directory_structure()
        
    elif choice == "4":
        install_requirements()
        
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()