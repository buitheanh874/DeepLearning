import os
from ultralytics import YOLO

# --- Configuration ---
# This path is derived from license_plate_detection.py,
# which prints "Model saved: runs/detect/license_plate_detector/weights/best.pt"
MODEL_PATH = 'best.pt'

# Confidence threshold for detection
CONF_THRESHOLD = 0.25
# --- End Configuration ---

def main():
    # 1. Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please ensure you have successfully trained the model")
        print("and that the MODEL_PATH variable in this script is correct.")
        return

    # 2. Load the trained model
    # We use the YOLO class from the ultralytics library to load the model
    print(f"Loading model from '{MODEL_PATH}'...")
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Loop to ask the user for image paths
    while True:
        image_path = input("\nEnter path to image for detection (or 'q' to quit): ").strip()

        # Exit if the user enters 'q'
        if image_path.lower() == 'q':
            print("Exiting...")
            break
            
        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at '{image_path}'. Please try again.")
            continue

        # 4. Run prediction on the image
        print(f"Running detection on '{image_path}'...")
        try:
            # The predict method is also used in license_plate_detection.py
            # save=True: Automatically saves the resulting image (with bounding boxes)
            #           to a new directory (usually 'runs/detect/predict/').
            # conf=...: Only keep detections with a confidence score higher than this.
            # verbose=False: Reduces console output.
            results = model.predict(
                source=image_path,
                conf=CONF_THRESHOLD,
                save=True,
                verbose=False
            )

            # Get the directory where the result was saved
            # results[0].save_dir contains the path (e.g., 'runs/detect/predict2')
            save_dir = results[0].save_dir
            saved_image_path = os.path.join(save_dir, os.path.basename(image_path))

            print("---")
            print("Detection complete!")
            print(f"Result image (with boxes) has been saved to: {saved_image_path}")
            print("---")

            # Optional: If you want the result image to pop up automatically,
            # uncomment the line below (you may need extra libraries)
            # results[0].show()

        except Exception as e:
            print(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()