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
    
    def train(self, data_yaml, epochs=50, imgsz=640, batch=16):
        """Train YOLOv8 model"""
        print(f"\n{'='*60}")
        print("TRAINING YOLOv8")
        print(f"{'='*60}")
        
        model_name = f'yolov8{self.model_size}.pt'
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        
        
        print(f"\nStarting training...")
        print(f"Epochs: {epochs} | Image size: {imgsz} | Batch: {batch}")
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='license_plate_detector',
            patience=20, 
            save=True,
            plots=True,
            device='cpu'  
        )
        
        print("\n✓ Training completed!")
        return results
    
    def validate(self):
        """Validate trained model"""
        print(f"\n{'='*60}")
        print("VALIDATION")
        print(f"{'='*60}")
        
        metrics = self.model.val()
        
        print(f"\nResults:")
        print(f"  mAP50:     {metrics.box.map50:.4f}")
        print(f"  mAP50-95:  {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.p[0]:.4f}")
        print(f"  Recall:    {metrics.box.r[0]:.4f}")
        
        return metrics
    
    def predict(self, image_path, conf=0.25, save=True):
        """Predict on single image"""
        print(f"\nPredicting: {image_path}")
        
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=save,
            show_labels=True,
            show_conf=True
        )
        
        for r in results:
            boxes = r.boxes
            print(f"\nDetected {len(boxes)} license plate(s):")
            for i, box in enumerate(boxes):
                conf = box.conf[0]
                cls = box.cls[0]
                xyxy = box.xyxy[0]
                print(f"  Plate {i+1}: confidence={conf:.3f}, bbox={xyxy.tolist()}")
        
        return results
    
    def predict_video(self, video_path, conf=0.25):
        """Predict on video"""
        print(f"\nProcessing video: {video_path}")
        
        results = self.model.predict(
            source=video_path,
            conf=conf,
            save=True,
            stream=True
        )
        
        for r in results:
            pass  
        
        print("✓ Video processing complete!")
    
    def export_model(self, format='onnx'):
        """Export model to different formats"""
        print(f"\nExporting model to {format}...")
        self.model.export(format=format)
        print(f"✓ Model exported!")


def create_demo_image(output_path='demo_car.jpg'):
    """Create demo car image for testing"""
    print(f"\nCreating demo image: {output_path}")
    
    img = np.random.randint(80, 120, (640, 640, 3), dtype=np.uint8)
    
    cv2.rectangle(img, (150, 200), (490, 450), (60, 70, 200), -1)
    cv2.rectangle(img, (180, 220), (460, 300), (120, 160, 220), -1)

    plate_x, plate_y = 270, 400
    plate_w, plate_h = 140, 45
    cv2.rectangle(img, (plate_x, plate_y), 
                 (plate_x + plate_w, plate_y + plate_h), (240, 240, 240), -1)
    cv2.rectangle(img, (plate_x, plate_y), 
                 (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 2)
    cv2.putText(img, '29A-12345', (plate_x + 15, plate_y + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"✓ Demo image created: {output_path}")
    return output_path


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("LICENSE PLATE DETECTION WITH YOLOv8")
    print("="*60)
    
    detector = YOLOLicensePlateDetector(model_size='n')
    
    data_yaml = detector.create_synthetic_dataset(num_images=500)
    
    results = detector.train(
        data_yaml=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16
    )
    
    metrics = detector.validate()
    
    demo_img = create_demo_image()
    detector.predict(demo_img, conf=0.25)
  
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved at: runs/detect/license_plate_detector/weights/best.pt")
    print(f"Results saved at: runs/detect/license_plate_detector/")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print("="*60 + "\n")  
    return detector


def load_and_predict(model_path, image_path):
    """Load trained model and predict"""
    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.25, save=True)
    return results


if __name__ == "__main__":
    try:
        detector = main()        
        print("✓ All tasks completed successfully!")     
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()