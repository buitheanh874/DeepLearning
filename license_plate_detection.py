from ultralytics import YOLO
import cv2
import numpy as np
import os
import yaml
from pathlib import Path
import shutil
import random  

class YOLOLicensePlateDetector:
    def __init__(self, model_size='n'):
        self.model_size = model_size
        self.model = None
        
    def create_synthetic_dataset(self, num_images=300, output_dir='synthetic_dataset'):
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
            
            car_colors = [[180, 60, 60], [60, 60, 180], [220, 220, 220], [40, 40, 40], [160, 160, 160]]
            car_color = car_colors[np.random.randint(0, len(car_colors))]
            
            cv2.rectangle(img, (car_x, car_y), (car_x+car_w, car_y+car_h), car_color, -1)
            
            win_y = car_y + 20
            win_h = int(car_h * 0.35)
            cv2.rectangle(img, (car_x+40, win_y), (car_x+car_w-40, win_y+win_h), [120, 160, 200], -1)
            
            plate_w = np.random.randint(120, 180)
            plate_h = int(plate_w * 0.35)
            plate_x = car_x + (car_w - plate_w) // 2 + np.random.randint(-30, 30)
            plate_y = car_y + car_h - plate_h - np.random.randint(10, 40)
            
            plate_x = max(10, min(plate_x, 630 - plate_w))
            plate_y = max(10, min(plate_y, 630 - plate_h))
            
            plate_bg = [[245, 245, 245], [240, 240, 120]][np.random.randint(0, 2)]
            
            cv2.rectangle(img, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h), plate_bg, -1)
            cv2.rectangle(img, (plate_x, plate_y), (plate_x+plate_w, plate_y+plate_h), [0, 0, 0], 3)
            
            cv2.putText(img, '29A-12345', (plate_x + 20, plate_y + plate_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            noise = np.random.normal(0, 15, img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            brightness = np.random.uniform(0.7, 1.4)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            cv2.imwrite(str(img_dir / f'car_{i:04d}.jpg'), img)
            
            x_center = (plate_x + plate_w / 2) / 640
            y_center = (plate_y + plate_h / 2) / 640
            norm_w = plate_w / 640
            norm_h = plate_h / 640
            
            with open(label_dir / f'car_{i:04d}.txt', 'w') as f:
                f.write(f'0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n')
    
    def prepare_custom_images(self, images_folder):
        images_path = Path(images_folder)
        if not images_path.exists():
            return None
        
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + list(images_path.glob('*.jpeg'))
        if not image_files:
            return None
        
        label_files = list(images_path.glob('*.txt'))
        if not label_files:
            return None
        
        output_dir = 'custom_dataset'
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
            return None
        
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
        
        return str(yaml_path)
    
    def combine_datasets(self, custom_yaml, synthetic_yaml, output_dir='final_dataset'):
        output_path = Path(output_dir)
        for split in ['train', 'val']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        custom_path = Path(custom_yaml).parent
        for split in ['train', 'val']:
            src_img = custom_path / split / 'images'
            src_lbl = custom_path / split / 'labels'
            dst_img = output_path / split / 'images'
            dst_lbl = output_path / split / 'labels'
            
            if src_img.exists():
                for f in src_img.glob('*.*'):
                    shutil.copy(f, dst_img / f"real_{f.name}")
                for f in src_lbl.glob('*.txt'):
                    shutil.copy(f, dst_lbl / f"real_{f.name}")
        
        synthetic_path = Path(synthetic_yaml).parent
        for split in ['train', 'val']:
            src_img = synthetic_path / split / 'images'
            src_lbl = synthetic_path / split / 'labels'
            dst_img = output_path / split / 'images'
            dst_lbl = output_path / split / 'labels'
            
            if src_img.exists():
                for f in src_img.glob('*.jpg'):
                    shutil.copy(f, dst_img / f"syn_{f.name}")
                for f in src_lbl.glob('*.txt'):
                    shutil.copy(f, dst_lbl / f"syn_{f.name}")
        
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
        
        return str(yaml_path)
    
    def train(self, data_yaml, epochs=50, imgsz=640, batch=16):
        model_name = f'yolov8{self.model_size}.pt'
        self.model = YOLO(model_name)
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='license_plate_detector',
            patience=20,
            save=True,
            plots=True,
            device=0,
            verbose=False,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.0
        )
        
        return results
    
    def validate(self):
        metrics = self.model.val(verbose=False)
        print(f"\nmAP50: {metrics.box.map50:.4f}")
        return metrics
     
    def visualize_results(self, test_images_folder='my_images', num_samples=6):
        import matplotlib.pyplot as plt
        images_path = Path(test_images_folder)
        image_files = list(images_path.glob('*.jpg'))[:num_samples]
        
        if not image_files:
            print("No images found for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, img_file in enumerate(image_files):
            results = self.model.predict(str(img_file), conf=0.25, verbose=False)
            
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'{img_file.name}', fontsize=10)
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
        print("Saved: prediction_results.png")
        plt.show()
    
    def plot_training_curves(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        
        results_path = Path('runs/detect/license_plate_detector/results.csv')
        if not results_path.exists():
            print("No training results found")
            return
        
        df = pd.read_csv(results_path)
        df.columns = df.columns.str.strip()  
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Box Loss')
        axes[0, 0].set_title('Bounding Box Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2)
        axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('Mean Average Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
        axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 1].plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
        axes[1, 1].plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', linewidth=2)
        axes[1, 1].plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Training Losses')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("Saved: training_curves.png")
        plt.show()
    def predict(self, image_path, conf=0.25, save=True):
        results = self.model.predict(source=image_path, conf=conf, save=save, verbose=False)
        return results

    def load_model(self, model_path):
        self.model = YOLO(model_path)


def main():
    detector = YOLOLicensePlateDetector(model_size='n')
    
    synthetic_yaml = detector.create_synthetic_dataset(num_images=300)
    
    images_folder = input("Images folder (default: my_images): ").strip()
    if not images_folder:
        images_folder = 'my_images'
    
    custom_yaml = detector.prepare_custom_images(images_folder)
    
    if custom_yaml:
        data_yaml = detector.combine_datasets(custom_yaml, synthetic_yaml)
    else:
        data_yaml = synthetic_yaml
    
    detector.train(data_yaml=data_yaml, epochs=50, batch=16)
    metrics = detector.validate()
    
    print(f"\nModel saved: runs/detect/license_plate_detector/weights/best.pt")
    return detector


if __name__ == "__main__":
    try:
        detector = main()
        print("Done")
    except Exception as e:
        print(f"Error: {e}")
