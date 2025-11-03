# License Plate Detection
AI model to detect license plates in images using YOLOv8.( just detect license plates!! not reading or doing  sth else)

## Technologies Used
-   **Programming Language:** Python
-   **Image Processing Library:** OpenCV
-   **Deep Learning Library:** TensorFlow / Keras
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
Enter: `my_images` → Wait for a super long time ( my cheap pc with i3-8100 CPU 16gb ram took about like 1-1,5 hours)

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
