import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import yaml
from pathlib import Path
import shutil

class YOLOLicensePlateDetector:
    def __init__(self, model_size='n'):
        """
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        Nano là nhẹ nhất, phù hợp bắt đầu
        """
        self.model_size = model_size
        self.model = None
        
    def create_synthetic_dataset(self, num_images=500, output_dir='license_plate_dataset'):
        """Tạo synthetic dataset với format YOLO"""
        print(f"\n{'='*60}")
        print("CREATING YOLO DATASET")
        print(f"{'='*60}")
        
      
        dataset_path = Path(output_dir)
        for split in ['train', 'val']:
            (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        train_count = int(num_images * 0.8)
        val_count = num_images - train_count
        
        print(f"\nGenerating {train_count} training images...")
        self._generate_images(train_count, dataset_path / 'train')
        
        
        print(f"Generating {val_count} validation images...")
        self._generate_images(val_count, dataset_path / 'val')
        
        
        data_yaml = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,  
            'names': ['license_plate']
        }
        
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"\n✓ Dataset created at: {dataset_path}")
        print(f"✓ Config file: {yaml_path}")
        return str(yaml_path)
     def _generate_images(self, num_images, split_path):
        """Generate synthetic car images with license plates"""
        img_dir = split_path / 'images'
        label_dir = split_path / 'labels'
        
        for i in range(num_images):
            img = np.random.randint(60, 100, (640, 640, 3), dtype=np.uint8)
            
            car_x = np.random.randint(100, 200)
            car_y = np.random.randint(150, 250)
            car_w = np.random.randint(300, 400)
            car_h = np.random.randint(200, 300)
            
            car_color = np.random.choice([
                [180, 60, 60],   
                [60, 60, 180],   
                [220, 220, 220], 
                [40, 40, 40],    
                [160, 160, 160]  
            ])
            cv2.rectangle(img, (car_x, car_y), (car_x+car_w, car_y+car_h), 
                         car_color.tolist(), -1)
            
            win_y = car_y + 20
            win_h = int(car_h * 0.35)
            cv2.rectangle(img, (car_x+40, win_y), (car_x+car_w-40, win_y+win_h),
                         [120, 160, 200], -1)
            
            plate_w = np.random.randint(120, 180)
            plate_h = int(plate_w * 0.35)
            plate_x = car_x + (car_w - plate_w) // 2 + np.random.randint(-30, 30)
            plate_y = car_y + car_h - plate_h - np.random.randint(10, 40)
            
            plate_x = max(10, min(plate_x, 630 - plate_w))
            plate_y = max(10, min(plate_y, 630 - plate_h))
            
            plate_bg = np.random.choice([[245, 245, 245], [240, 240, 120]])
            cv2.rectangle(img, (plate_x, plate_y), 
                         (plate_x+plate_w, plate_y+plate_h), 
                         plate_bg.tolist(), -1)
            
            cv2.rectangle(img, (plate_x, plate_y), 
                         (plate_x+plate_w, plate_y+plate_h), 
                         [0, 0, 0], 3)
            
            cv2.putText(img, '29A-12345', 
                       (plate_x + 20, plate_y + plate_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            noise = np.random.normal(0, 15, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            brightness = np.random.uniform(0.7, 1.4)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            
            img_path = img_dir / f'car_{i:04d}.jpg'
            cv2.imwrite(str(img_path), img)
            
            x_center = (plate_x + plate_w / 2) / 640
            y_center = (plate_y + plate_h / 2) / 640
            norm_w = plate_w / 640
            norm_h = plate_h / 640
            
            label_path = label_dir / f'car_{i:04d}.txt'
            with open(label_path, 'w') as f:
                f.write(f'0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n')
            
            if (i + 1) % 100 == 0:
                print(f"  ✓ Generated {i + 1}/{num_images}")
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        print(f"\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        os.makedirs('models', exist_ok=True)
        
        callbacks = [
            ModelCheckpoint('models/best_model.h5', monitor='val_loss', 
                          save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, 
                         restore_best_weights=True, verbose=1)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def plot_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("Training history saved as 'training_history.png'")
        plt.show()
    
    def save_model(self, filepath='models/final_model.h5'):
        self.model.save(filepath)
        print(f"Model saved: {filepath}")


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")
    
    predictions = model.predict(X_test, verbose=0)
    
    ious = []
    for pred, true in zip(predictions, y_test):
        iou = calculate_iou(pred, true)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Min IoU: {np.min(ious):.4f}")
    print(f"Max IoU: {np.max(ious):.4f}")
    print(f"Median IoU: {np.median(ious):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(mean_iou, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_iou:.4f}')
    plt.xlabel('IoU Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('IoU Distribution on Test Set', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('iou_distribution.png', dpi=150, bbox_inches='tight')
    print("IoU distribution saved as 'iou_distribution.png'")
    plt.show()
    
    return mean_iou, ious


def visualize_predictions(model, X_test, y_test, num_samples=6):
    print(f"\nVisualizing {num_samples} predictions...")
    
    predictions = model.predict(X_test[:num_samples], verbose=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = (X_test[i] * 255).astype(np.uint8).copy()
        pred_box = predictions[i]
        true_box = y_test[i]
        
        h, w = img.shape[:2]
        
        x, y, bw, bh = true_box
        cv2.rectangle(img, (int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h)),
                     (0, 255, 0), 2)
        
        x, y, bw, bh = pred_box
        cv2.rectangle(img, (int(x*w), int(y*h)), (int((x+bw)*w), int((y+bh)*h)),
                     (255, 0, 0), 2)
        
        iou = calculate_iou(pred_box, true_box)
        
        axes[i].imshow(img)
        axes[i].set_title(f'IoU: {iou:.3f}', fontsize=12)
        axes[i].axis('off')
        axes[i].text(10, 20, 'Green: GT', color='green', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[i].text(10, 40, 'Red: Pred', color='red', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    print("Predictions saved as 'predictions_visualization.png'")
    plt.show()


def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=3):
    h, w = image.shape[:2]
    x, y, box_w, box_h = bbox
    
    x1 = int(x * w)
    y1 = int(y * h)
    x2 = int((x + box_w) * w)
    y2 = int((y + box_h) * h)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    label_size = cv2.getTextSize('License Plate', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                 (x1 + label_size[0], y1), color, -1)
    cv2.putText(image, 'License Plate', (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image


def predict_image(model, image_path):
    print(f"\nProcessing: {image_path}")
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()
    
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized.astype('float32') / 255.0
    
    prediction = model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)[0]
    
    result = draw_bounding_box(original, prediction)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title('Detected License Plate')
    plt.axis('off')
    plt.show()
    
    print(f"Predicted Bounding Box:")
    print(f"  x: {prediction[0]:.4f}")
    print(f"  y: {prediction[1]:.4f}")
    print(f"  width: {prediction[2]:.4f}")
    print(f"  height: {prediction[3]:.4f}")
    
    return prediction


def create_demo_image(output_path='demo_input.jpg'):
    print(f"\nCreating demo image: {output_path}")
    
    img = np.random.randint(80, 150, (480, 640, 3), dtype=np.uint8)
    
    cv2.rectangle(img, (100, 150), (540, 400), (100, 120, 140), -1)
    cv2.rectangle(img, (150, 200), (250, 350), (60, 70, 80), -1)
    cv2.rectangle(img, (390, 200), (490, 350), (60, 70, 80), -1)
    
    plate_x, plate_y = 250, 320
    plate_w, plate_h = 140, 40
    cv2.rectangle(img, (plate_x, plate_y), 
                 (plate_x + plate_w, plate_y + plate_h), (230, 230, 230), -1)
    cv2.rectangle(img, (plate_x, plate_y), 
                 (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 2)
    cv2.putText(img, '29A-12345', (plate_x + 15, plate_y + 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Demo image created: {output_path}")
    
    return output_path


def main():
    print("=" * 70)
    print("VEHICLE LICENSE PLATE LOCALIZATION")
    print("=" * 70)
    
    detector = LicensePlateDetector()
    detector.build_model()
    
    X, y = detector.create_dataset(num_samples=1000)
    
    split_train = int(0.7 * len(X))
    split_val = int(0.85 * len(X))
    
    X_train, y_train = X[:split_train], y[:split_train]
    X_val, y_val = X[split_train:split_val], y[split_train:split_val]
    X_test, y_test = X[split_val:], y[split_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    history = detector.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    
    detector.plot_history(history)
    
    mean_iou, ious = evaluate_model(detector.model, X_test, y_test)
    
    visualize_predictions(detector.model, X_test, y_test)
    
    detector.save_model('models/final_model.h5')
    
    demo_img = create_demo_image()
    predict_image(detector.model, demo_img)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Best Val Loss: {min(history.history['val_loss']):.6f}")


if __name__ == "__main__":
    main()