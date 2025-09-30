# Pothole Detection ML Model 🕳️🛣️

A machine learning project for detecting potholes in road images using Convolutional Neural Networks (CNN) with transfer learning. This system can identify potholes in real-time from camera feeds and maintain a local database for counting and tracking.

## 🚀 Features

- **CNN-based Classification**: Uses pre-trained models (MobileNetV2, ResNet50, EfficientNetB0) with transfer learning
- **Data Augmentation**: Comprehensive image augmentation for robust training
- **Real-time Detection**: Can process camera feeds for live pothole detection
- **Local Database**: SQLite database for storing pothole counts and locations
- **Multiple Model Options**: Choose from different base architectures based on speed/accuracy requirements
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations

## 📁 Project Structure

```
PotHoleRecognition/
├── data/                          # Data directory
│   ├── raw/                       # Raw images
│   │   ├── pothole/              # Pothole images
│   │   └── normal/               # Normal road images
│   └── processed/                # Processed & split data
│       ├── train/
│       ├── val/
│       └── test/
├── models/                        # Saved models
├── results/                       # Training results & plots
├── logs/                         # Training logs
├── setup.py                     # Environment setup
├── data_downloader.py           # Dataset download utilities
├── data_preprocessing.py        # Data processing & augmentation
├── train_model.py              # Model training & evaluation
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## ⚡ Quick Start

### Option 1: Full Setup with Real Data

1. **Setup Environment**
   ```bash
   python setup.py
   # Choose option 1 for full setup
   ```

2. **Get Dataset** (Choose one):
   
   **Option A: Download from Kaggle**
   ```bash
   # Setup Kaggle API first: https://www.kaggle.com/docs/api
   python data_downloader.py
   # Choose option 3 and enter: andrewmvd/pothole-detection-dataset
   ```
   
   **Option B: Manual Collection**
   - Add pothole images to `data/raw/pothole/`
   - Add normal road images to `data/raw/normal/`
   
   **Option C: Use Web Datasets**
   ```bash
   python data_downloader.py
   # Choose option 2 for dataset sources info
   ```

3. **Process Data**
   ```bash
   python data_preprocessing.py
   # Choose option 2 to process your data
   ```

4. **Train Model**
   ```bash
   python train_model.py
   # Choose option 1 to train new model
   ```

### Option 2: Quick Demo with Synthetic Data

```bash
python setup.py
# Choose option 2 for quick start demo
```

This will create synthetic data and train a demo model in minutes!

## 🛠️ Manual Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

**For Classification Task:**
- Place pothole images in: `data/raw/pothole/`
- Place normal road images in: `data/raw/normal/`

**Supported Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`

### 3. Process Data

```bash
python data_preprocessing.py
```

Choose from:
- Option 1: Create synthetic data for demo
- Option 2: Process your real images
- Option 3: Organize existing raw data

### 4. Train Model

```bash
python train_model.py
```

Training options:
- **Base Models**: MobileNetV2 (fast), ResNet50 (accurate), EfficientNetB0 (balanced)
- **Epochs**: Recommended 30-50 for real data, 5-10 for demo
- **Batch Size**: 16-32 depending on GPU memory

## 📊 Model Performance

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Ratio of correctly predicted potholes
- **Recall**: Ratio of actual potholes detected
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## 🔧 Configuration

Modify `config.json` to customize:

```json
{
  "model": {
    "img_size": [224, 224],
    "base_model": "MobileNetV2",
    "num_classes": 2
  },
  "training": {
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "data": {
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1
  }
}
```

## 📈 Usage Examples

### Train a New Model

```python
from train_model import PotholeDetectionModel

model = PotholeDetectionModel(img_size=(224, 224))
history = model.train_model(
    epochs=30, 
    batch_size=32, 
    base_model='MobileNetV2'
)
```

### Predict Single Image

```python
model = PotholeDetectionModel()
result = model.predict_image(
    image_path='test_image.jpg',
    model_path='models/best_model.h5'
)

print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Evaluate Existing Model

```python
model = PotholeDetectionModel()
report, cm = model.evaluate_model('models/best_model.h5')
```

## 🎯 Recommended Datasets

### Public Datasets:

1. **Kaggle Datasets:**
   - `andrewmvd/pothole-detection-dataset`
   - `sachinpatel21/pothole-image-dataset`
   - `chitholian/annotated-potholes-dataset`

2. **Roboflow Universe:**
   - Search for "pothole detection" datasets
   - Many annotated datasets available for free

3. **Academic Datasets:**
   - Road Damage Dataset (RDD)
   - CrackForest Dataset

### Data Collection Tips:

- **Variety**: Different lighting conditions, weather, road types
- **Balance**: Equal number of pothole and normal road images
- **Quality**: Clear, focused images with visible potholes
- **Resolution**: At least 224x224 pixels, higher is better
- **Angles**: Various camera angles and distances

## 🏗️ Next Steps (Real-time Application)

After training your model, you can extend this project to:

1. **Real-time Camera Detection**
2. **Mobile App Integration** 
3. **GPS Location Tracking**
4. **Database Integration**
5. **Web Dashboard**
6. **API Development**

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- OpenCV 4.8+
- 4GB+ RAM
- GPU recommended for faster training

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named cv2"**
   ```bash
   pip install opencv-python
   ```

2. **TensorFlow GPU issues**
   ```bash
   pip install tensorflow[and-cuda]
   ```

3. **Low accuracy with synthetic data**
   - Use real pothole images for better performance
   - Increase training epochs
   - Try different base models

4. **Out of memory errors**
   - Reduce batch size
   - Reduce image size
   - Use MobileNetV2 instead of ResNet50

### Performance Tips:

- **For Speed**: Use MobileNetV2 with smaller images (224x224)
- **For Accuracy**: Use ResNet50 or EfficientNetB0 with larger images
- **For Balance**: Use EfficientNetB0 with standard settings

## 📄 License

This project is open source. Feel free to use and modify for your needs.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Submit pull request

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all requirements are installed
3. Ensure proper data structure
4. Check Python version compatibility

---

**Happy Pothole Detecting! 🛣️✨**