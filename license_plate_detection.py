import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

class LicensePlateDetector:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        print("Building model...")
        
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(4, activation='sigmoid', name='bbox_output')(x)
        
        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        print(f"Model created with {self.model.count_params():,} parameters")
        return self.model
    
    def create_dataset(self, num_samples=1000):
        print(f"Creating dataset with {num_samples} samples...")
        
        X, y = [], []
        
        for i in range(num_samples):
            img = np.random.randint(50, 200, self.input_shape, dtype=np.uint8)
            
            for _ in range(3):
                rx = np.random.randint(0, 180)
                ry = np.random.randint(0, 180)
                rw = np.random.randint(20, 60)
                rh = np.random.randint(20, 60)
                color = np.random.randint(0, 255, 3)
                img[ry:ry+rh, rx:rx+rw] = color
            
            plate_w = np.random.randint(60, 100)
            plate_h = int(plate_w * 0.3)
            plate_x = np.random.randint(20, 224 - plate_w - 20)
            plate_y = np.random.randint(100, 224 - plate_h - 20)
            
            img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = [240, 240, 240]
            
            noise = np.random.randint(-20, 20, (plate_h, plate_w, 3))
            img[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] += noise
            img = np.clip(img, 0, 255)
            
            img_normalized = img.astype('float32') / 255.0
            X.append(img_normalized)
            
            bbox = [plate_x / 224.0, plate_y / 224.0, plate_w / 224.0, plate_h / 224.0]
            y.append(bbox)
            
            if (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        return np.array(X), np.array(y)
    
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