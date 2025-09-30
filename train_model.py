import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from pathlib import Path
import cv2
from datetime import datetime

class PotholeDetectionModel:
    """
    CNN-based pothole detection model using transfer learning
    """
    
    def __init__(self, img_size=(224, 224), num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.data_dir = Path("data")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Class names
        self.class_names = ['normal', 'pothole']
        
    def create_data_generators(self, batch_size=32):
        """
        Create data generators for training, validation, and testing
        """
        print("Creating data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir / 'processed' / 'train',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'processed' / 'val',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.data_dir / 'processed' / 'test',
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def build_model(self, base_model_name='MobileNetV2', fine_tune=False):
        """
        Build CNN model using transfer learning
        """
        print(f"Building model with {base_model_name}...")
        
        # Load pre-trained base model
        if base_model_name == 'MobileNetV2':
            base_model = applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'ResNet50':
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers initially
        base_model.trainable = fine_tune
        
        # Build complete model
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing
        x = tf.cast(inputs, tf.float32)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom top layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        return self.model
    
    def setup_callbacks(self):
        """
        Setup training callbacks
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        callbacks = [
            ModelCheckpoint(
                filepath=self.model_dir / f'best_pothole_model_{timestamp}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, epochs=50, batch_size=32, base_model='MobileNetV2'):
        """
        Train the pothole detection model
        """
        print("=== STARTING TRAINING ===")
        
        # Create data generators
        train_gen, val_gen, test_gen = self.create_data_generators(batch_size)
        
        # Build model
        self.build_model(base_model)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(test_gen, verbose=1)
        
        print(f"\nTest Results:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = self.model_dir / f'pothole_model_final_{timestamp}.h5'
        self.model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        return self.history
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, model_path=None):
        """
        Evaluate model and generate detailed metrics
        """
        if model_path:
            self.model = keras.models.load_model(model_path)
        
        if self.model is None:
            print("No model available. Train or load a model first.")
            return
        
        # Create test generator
        _, _, test_gen = self.create_data_generators()
        
        # Predictions
        print("Making predictions...")
        predictions = self.model.predict(test_gen)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # True labels
        true_classes = test_gen.classes
        
        # Classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.model_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report, cm
    
    def predict_image(self, image_path, model_path=None):
        """
        Predict pothole for a single image
        """
        if model_path:
            self.model = keras.models.load_model(model_path)
        
        if self.model is None:
            print("No model available. Train or load a model first.")
            return None
        
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, self.img_size)
        image_normalized = image_resized / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image_batch)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        result = {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'is_pothole': predicted_class == 1,
            'probabilities': {
                'normal': float(prediction[0][0]),
                'pothole': float(prediction[0][1])
            }
        }
        
        # Display result
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        colors = ['green' if predicted_class == 0 else 'red']
        bars = plt.bar(self.class_names, prediction[0], color=colors, alpha=0.7)
        plt.title(f'Prediction: {self.class_names[predicted_class]} ({confidence:.2%})')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        
        # Add confidence text on bars
        for bar, prob in zip(bars, prediction[0]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return result

def main():
    """
    Main training script
    """
    print("=== POTHOLE DETECTION MODEL TRAINING ===")
    
    # Initialize model
    model = PotholeDetectionModel(img_size=(224, 224))
    
    print("Choose an option:")
    print("1. Train new model")
    print("2. Evaluate existing model")
    print("3. Predict single image")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nChoose base model:")
        print("1. MobileNetV2 (faster, smaller)")
        print("2. ResNet50 (more accurate)")
        print("3. EfficientNetB0 (balanced)")
        
        base_choice = input("Enter choice (1-3): ").strip()
        base_models = {'1': 'MobileNetV2', '2': 'ResNet50', '3': 'EfficientNetB0'}
        base_model = base_models.get(base_choice, 'MobileNetV2')
        
        epochs = int(input("Enter number of epochs (default 30): ").strip() or "30")
        batch_size = int(input("Enter batch size (default 32): ").strip() or "32")
        
        # Train model
        history = model.train_model(epochs=epochs, batch_size=batch_size, base_model=base_model)
        
        # Plot training history
        model.plot_training_history()
        
        # Evaluate model
        model.evaluate_model()
        
    elif choice == "2":
        model_path = input("Enter model path (leave empty for latest): ").strip()
        if not model_path:
            # Find latest model
            model_files = list(model.model_dir.glob("*.h5"))
            if model_files:
                model_path = str(max(model_files, key=os.path.getctime))
                print(f"Using model: {model_path}")
            else:
                print("No model files found!")
                return
        
        model.evaluate_model(model_path)
        
    elif choice == "3":
        model_path = input("Enter model path (leave empty for latest): ").strip()
        if not model_path:
            model_files = list(model.model_dir.glob("*.h5"))
            if model_files:
                model_path = str(max(model_files, key=os.path.getctime))
                print(f"Using model: {model_path}")
            else:
                print("No model files found!")
                return
        
        image_path = input("Enter image path: ").strip()
        result = model.predict_image(image_path, model_path)
        if result:
            print(f"\nPrediction Result:")
            print(f"Class: {result['class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Is Pothole: {result['is_pothole']}")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()