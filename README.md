# License Plate Detection
AI model to detect license plates in images using YOLOv8.( just detect license plates!! not reading or doing  sth else)
## Business requirement
- **object:** the system that automatically detect the license plates in the image ( first step to take and read all the license plate in the future)
-  **Usecase:** the camera record the vehiclies -> the system detect and take the position of the license plates -> store the resuld 
## Requirement Specification
- **the accuracy:** mAP > 90%
- **image process time < 1s**
- **able to deal with variable light conditions**
  
  ```mermaid
erDiagram
    ScanLog {
        int scan_id PK "Primary Key"
        string image_name "File name of the scanned image"
        datetime scanned_at "Timestamp of the scan"
    }

    DetectedPlate {
        int plate_id PK "Primary Key"
        int scan_id FK "Foreign Key to ScanLog"
        float confidence "Confidence score (e.g., 0.95)"
        string coordinates "Bounding box (e.g., [x, y, w, h])"
    }

    ScanLog ||--|{ DetectedPlate : "has"

  ```


## Technologies Used
-   **Programming Language:** Python
-   **Image Processing Library:** OpenCV
-   **Deep Learning Library:** PyTorch, Ultralytics
-   **Algorithms:** YOLOv8, CNN



## Installation and Usage

**Clone the repository:**
```bash
git clone https://github.com/buitheanh874/DeepLearning
cd license-plate-recognition
```

**Prepare Images**
- Create folder `my_images/`
- Add car,motorbike,... images (at least 50 real photos for better performence)

 ->> HOW TO label images by your self : -> go:https://www.makesense.ai/ -> put image in -> create label and name it = License_plate
                                                                        -> draw a box around the license plate -> extract -> copy all of the them into my_images

**Install the required libraries:**
```bash
pip install -r requirements.txt
```

**Run detection on an image:**
```bash
python license_plate_detection.py
```
Enter: `my_images` → Wait for a super long time ( my cheap pc with i3-8100 CPU 16gb ram took about like 1-1,5 hours)  ->> solution : run on google colab
---->> If your computer is strong enough to run and you really want to run by yourself -> device = 0 -> device = "cpu"

**Project Structure:**
```
DeepLearning/
├── license_plate_detection.py
├── requirements.txt
└── my_images/
    ├── car1.jpg
    ├── car1.txt  (the position with yolo form tell where the label is)
    └── bike1.jpg
```

## Features
- Synthetic data generation (300 images)
- Data augmentation (rotation, flip, color)
- Real + synthetic training
- Easy to use
