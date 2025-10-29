from ultralytics import YOLO
import cv2
import numpy as np
import os
import yaml
from pathlib import Path
import shutil

class YOLOLicensePlateDetector:
    def __init__(self, model_size='n'):
        self.model_size = model_size
        self.model = None
        self.base_model_path = None
        
    def create_synthetic_dataset(self, num_images=300, output_dir='synthetic_dataset'):
        """Tạo synthetic dataset"""
        print(f"Creating synthetic dataset ({num_images} images)...")
        
        dataset_path = Path(output_dir)
        for split in ['train', 'val']:
            (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        train_count = int(num_images * 0.8)
        val_count = num_images - train_count
        
        self._generate_images(train_count, dataset_path / 'train')
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
        
        print(f"✓ Synthetic: {train_count} train, {val_count} val")
        return str(yaml_path)
    
    def _generate_images(self, num_images, split_path):
        img_dir = split_path / 'images'
        label_dir = split_path / 'labels'
        
        for i in range(num_images):
            img = np.random.randint(60, 100, (640, 640, 3), dtype=np.uint8)
            
            car_x = np.random.randint(100, 200)
            car_y = np.random.randint(150, 250)
            car_w = np.random.randint(300, 400)
            car_h = np.random.randint(200, 300)
            
            car_color = np.random.choice([
                [180, 60, 60], [60, 60, 180], [220, 220, 220],
                [40, 40, 40], [160, 160, 160]
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
    
    def prepare_real_dataset(self, images_folder):
        """Chuẩn bị dataset từ ảnh thật của bạn"""
        print(f"Loading custom images from: {images_folder}")
        
        images_path = Path(images_folder)
        if not images_path.exists():
            print(f"✗ Folder not found: {images_folder}")
            return None
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + list(images_path.glob('*.jpeg'))
        
        if not image_files:
            print(f"✗ No images found")
            return None
        
        label_files = list(images_path.glob('*.txt'))
        
        if not label_files:
            return None
        
        output_dir = 'real_dataset'
        output_path = Path(output_dir)
        
        for split in ['train', 'val']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        paired_data = []
        for img_file in image_files:
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                paired_data.append((img_file, label_file))
        
        if not paired_data:
            print("✗ No valid image-label pairs")
            return None
        
        import random
        random.shuffle(paired_data)
        split_idx = int(len(paired_data) * 0.8)
        
        train_data = paired_data[:split_idx]
        val_data = paired_data[split_idx:]
        
        for img, lbl in train_data:
            shutil.copy(img, output_path / 'train' / 'images' / img.name)
            shutil.copy(lbl, output_path / 'train' / 'labels' / lbl.name)
        
        for img, lbl in val_data:
            shutil.copy(img, output_path / 'val' / 'images' / img.name)
            shutil.copy(lbl, output_path / 'val' / 'labels' / lbl.name)
        
        data_yaml = {
            'path': str(output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 1,
            'names': ['license_plate']
        }
        
        yaml_path = output_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"✓ Custom images: {len(train_data)} train, {len(val_data)} val")
        return str(yaml_path)
    
    def train_base_model(self, data_yaml, epochs=30, imgsz=640, batch=16):
        """Train base model với synthetic data"""
        print(f"\n{'='*60}")
        print("PHASE 1: Training BASE MODEL with Synthetic Data")
        print(f"{'='*60}")
    
        model_name = f'yolov8{self.model_size}.pt'
        self.model = YOLO(model_name)
    
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='base_model_synthetic',
            patience=15,
            save=True,
            plots=True,
            device='cpu',
            verbose=False
        )
    
        # Lưu đường dẫn base model
        self.base_model_path = 'runs/detect/base_model_synthetic/weights/best.pt'
    
        print("✓ Base model training completed")
        print(f"✓ Model saved: {self.base_model_path}")
    
        return results

    def finetune_with_real_data(self, data_yaml, epochs=30, imgsz=640, batch=16):
        """Fine-tune với real data"""
        print(f"\n{'='*60}")
        print("PHASE 2: FINE-TUNING with Real Data")
        print(f"{'='*60}")
    
        if self.base_model_path is None or not Path(self.base_model_path).exists():
            print("✗ Base model not found. Training from scratch instead...")
            self.model = YOLO(f'yolov8{self.model_size}.pt')
        else:
            print(f"Loading base model from: {self.base_model_path}")
            self.model = YOLO(self.base_model_path)
    
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='finetuned_model_real',
            patience=20,
            save=True,
            plots=True,
            device='cpu',
            verbose=False,
            lr0=0.001  # ← Learning rate thấp cho fine-tuning
        )
    
        print("✓ Fine-tuning completed")
    
        return results
    
    def validate(self):
        """Validate model"""
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
    
        metrics = self.model.val(verbose=False)
    
        print(f"mAP50:     {metrics.box.map50:.4f}")
        print(f"mAP50-95:  {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.p[0]:.4f}")
        print(f"Recall:    {metrics.box.r[0]:.4f}")
        print("="*60)
    
        return metrics
    
    def predict(self, image_path, conf=0.25, save=True):
        """Predict on new images"""
        print(f"\nPredicting on: {image_path}")
    
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save,
            verbose=False
        )
    
        for r in results:
            if len(r.boxes) > 0:
                print(f"✓ Detected {len(r.boxes)} license plate(s)")
            else:
                print("✗ No license plates detected")
    
        return results

def main():
    """Auto train with synthetic + your real images"""
    print("\n" + "="*60)
    print("LICENSE PLATE DETECTION - AUTO TRAINING")
    print("="*60)
    
    detector = YOLOLicensePlateDetector(model_size='n')
    
    print("\n[1/3] Creating synthetic dataset...")
    synthetic_yaml = detector.create_synthetic_dataset(num_images=300)
    
    print("\n[2/3] Loading your real images...")
    images_folder = input("Path to your images folder (default: my_images): ").strip()
    if not images_folder:
        images_folder = 'my_images'
    
    custom_yaml = detector.prepare_custom_images(images_folder)
    
    if custom_yaml:
        data_yaml = detector.combine_datasets(custom_yaml, synthetic_yaml)
    else:
        print("\n⚠ Using synthetic only (no custom images found)")
        data_yaml = synthetic_yaml
    
    print("\n[3/3] Training model...")
    detector.train(data_yaml=data_yaml, epochs=50, batch=16)
    
    metrics = detector.validate()
    
    print(f"\n{'='*60}")
    print("✓ DONE")
    print(f"{'='*60}")
    print(f"Model: runs/detect/license_plate_detector/weights/best.pt")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print("="*60 + "\n")
    
    return detector


if __name__ == "__main__":
    try:
        detector = main()
        print("✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


